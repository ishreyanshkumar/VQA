import streamlit as st
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, ViTModel, AutoTokenizer, ViTImageProcessor
from peft import PeftModel

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Qwen2.5-VQA", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stImage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIGURATION
# ==========================================
VIT_ID = "google/vit-base-patch32-384"
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 3. MODEL DEFINITION (Cached)
# ==========================================
class VQAModel(nn.Module):
    def __init__(self, lora_path="output/final_lora_model", conn_path="output/final_connector.pth"):
        super().__init__()
        
        # Path Validation
        if not os.path.exists(lora_path):
            st.error(f"⚠️ LoRA path '{lora_path}' not found. Have you run train.py?")
        if not os.path.exists(conn_path):
            st.error(f"⚠️ Connector path '{conn_path}' not found.")

        # 1. Load Tokenizer & Processor
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
        self.processor = ViTImageProcessor.from_pretrained(VIT_ID)

        # 2. Load Vision Encoder (Frozen)
        self.vit = ViTModel.from_pretrained(VIT_ID).to(DEVICE).eval()
        
        # 3. Load LLM + LoRA Adapters
        base_llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            attn_implementation="sdpa"
        ).to(DEVICE)
        
        self.llm = PeftModel.from_pretrained(base_llm, lora_path).eval()
        
        # 4. Load Connector
        vis_dim = self.vit.config.hidden_size 
        llm_dim = self.llm.config.hidden_size 
        
        self.connector = nn.Sequential(
            nn.Linear(vis_dim, llm_dim), 
            nn.GELU(), 
            nn.Linear(llm_dim, llm_dim)
        ).to(DEVICE)
        
        # Load weights safely
        try:
            self.connector.load_state_dict(torch.load(conn_path, map_location=DEVICE))
        except Exception as e:
            st.error(f"❌ Error loading connector weights: {e}")

    def process_image(self, image_input):
        """Processes PIL Image or URL to tensor"""
        try:
            if isinstance(image_input, str) and image_input.startswith('http'):
                response = requests.get(image_input)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img = np.array(img)
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input.convert('RGB'))
            else:
                return None

            # Smart Resize (Pad to Square)
            h, w = img.shape[:2]
            scale = IMG_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = img
            
            return self.processor(canvas, return_tensors="pt").pixel_values.to(DEVICE)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

    def generate_answer(self, image_tensor, history, max_tokens=200):
        # 1. Get Visual Embeddings
        with torch.no_grad():
            vit_out = self.vit(image_tensor).last_hidden_state
            img_embeds = self.connector(vit_out.to(dtype=torch.float32)).to(dtype=torch.float16)

        # 2. Build Text Prompt from History
        # Format: System -> User (First) -> Assistant -> User -> ...
        text_prompt = "<|im_start|>system\nYou are a helpful visual assistant. You can see the image provided by the user.<|im_end|>\n"
        
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            text_prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
        
        # We append the "start" of the assistant's next turn for generation
        text_prompt += "<|im_start|>assistant\n"
        
        text_inputs = self.tokenizer(text_prompt, return_tensors="pt").to(DEVICE)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        # 3. Concatenate (Image Embeds + Text Embeds)
        # We prepend image embeds to the ENTIRE conversation. 
        # Ideally, image embeds go before the first user question, but this simple prepend works well for VQA context.
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat((img_embeds, text_embeds), dim=1)
        
        # Attention Mask
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=DEVICE)
        combined_mask = torch.cat((img_mask, attention_mask), dim=1)

        # 4. Generate
        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )

        # Decode response
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.strip()

@st.cache_resource
def get_model():
    return VQAModel()

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
def main():
    # Load Model (Cached)
    with st.spinner("🧠 Loading Qwen2.5-VQA Model... (This happens only once)"):
        model = get_model()

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_image_tensor" not in st.session_state:
        st.session_state.current_image_tensor = None
    if "image_file" not in st.session_state:
        st.session_state.image_file = None

    # --- SIDEBAR: Image Upload ---
    with st.sidebar:
        st.title("🖼️ Visual Input")
        st.write("Upload an image to start chatting.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        url_input = st.text_input("Or paste an Image URL:")
        
        # Handle Input
        image_source = None
        if uploaded_file is not None:
            image_source = Image.open(uploaded_file)
            st.session_state.image_file = image_source
        elif url_input:
            image_source = url_input
            st.session_state.image_file = image_source # We'll load the preview later if needed

        # Process Button
        if st.button("Load Image"):
            if image_source:
                with st.spinner("Encoding image..."):
                    tensor = model.process_image(image_source)
                    if tensor is not None:
                        st.session_state.current_image_tensor = tensor
                        # Reset chat history on new image
                        st.session_state.messages = [] 
                        st.success("Image Loaded! Start chatting.")
            else:
                st.warning("Please upload an image or provide a URL.")
        
        # Show Current Image Preview
        if st.session_state.image_file:
            st.image(st.session_state.image_file, caption="Current Context", use_container_width=True)

    # --- MAIN CHAT AREA ---
    st.title("👁️ Qwen2.5-VQA Chat")
    st.caption("Powered by ViT + Qwen2.5-3B + LoRA")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the image..."):
        # Guard: Check if image is loaded
        if st.session_state.current_image_tensor is None:
            st.error("Please load an image in the sidebar first!")
        else:
            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = model.generate_answer(
                            st.session_state.current_image_tensor,
                            st.session_state.messages
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()