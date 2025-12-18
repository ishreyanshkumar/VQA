import os
import cv2
import json
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    ViTModel, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    AutoTokenizer, 
    ViTImageProcessor,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Adjust these paths to your local environment
    LLAVA_JSON = "./data/llava_instruct_150k.json"
    COCO_TRAIN_DIR = "./data/coco2017/train2017"
    OUTPUT_DIR = "./output"
    
    VIT_ID = "google/vit-base-patch32-384"
    LLM_ID = "Qwen/Qwen2.5-3B-Instruct" 
    
    MAX_LEN = 128
    IMG_SIZE = 384  
    MAX_SAMPLES = 200000
    BATCH_SIZE = 4       
    GRAD_ACCUM = 16       
    EPOCHS = 1 
    LR = 5e-5 

config = Config()

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class QwenVQA(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Vision Encoder
        print(f"Loading Vision Encoder ({config.VIT_ID})...")
        self.vit = ViTModel.from_pretrained(config.VIT_ID)
        for p in self.vit.parameters(): p.requires_grad = False
            
        # 2. LLM
        print("Loading Qwen LLM...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_ID, 
            torch_dtype=torch.float16, 
            trust_remote_code=True, 
            attn_implementation="sdpa" 
        )
        self.llm.config.use_cache = False
        
        # 3. LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=64,           
            lora_alpha=128, 
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
        )
        self.llm = get_peft_model(self.llm, peft_config)
        
        # 4. Connector
        vis_dim = self.vit.config.hidden_size 
        llm_dim = self.llm.config.hidden_size 
        self.connector = nn.Sequential(
            nn.Linear(vis_dim, llm_dim), 
            nn.GELU(), 
            nn.Linear(llm_dim, llm_dim)
        ).to(torch.float32)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Vision Pass
        with torch.no_grad():
            vit_out = self.vit(pixel_values).last_hidden_state
        
        # Projection
        img_embeds = self.connector(vit_out.to(torch.float32))
        
        # Text Embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Concatenate (Image + Text)
        combined_embeds = torch.cat((img_embeds.to(text_embeds.dtype), text_embeds), dim=1)
        
        # Attention Mask Handling
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=img_embeds.device)
        combined_mask = torch.cat((img_mask, attention_mask), dim=1)
        
        if labels is not None:
            # Mask labels for the image part so we calculate loss only on text
            img_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=labels.device)
            combined_labels = torch.cat((img_labels, labels), dim=1)
            return self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=combined_labels)
        else:
            return self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask)

# ==========================================
# 3. DATASET
# ==========================================
processor = ViTImageProcessor.from_pretrained(config.VIT_ID)
tokenizer = AutoTokenizer.from_pretrained(config.LLM_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

class HybridTrainDataset(Dataset):
    def __init__(self, llava_json, coco_dir, max_samples=200000):
        self.samples = []
        
        # 1. LLaVA
        if os.path.exists(llava_json):
            print(f"--> Loading LLaVA metadata...")
            try:
                with open(llava_json, 'r') as f: raw = json.load(f)
                for item in raw:
                    if 'image' in item:
                        self.samples.append({
                            "type": "llava", 
                            "path": os.path.join(coco_dir, item['image']), 
                            "conv": item['conversations']
                        })
            except: pass
        
        # 2. TextVQA & VizWiz (HuggingFace)
        print("--> Loading TextVQA & VizWiz metadata...")
        try:
            tvqa = load_dataset("textvqa", split="train")
            for item in tvqa:
                self.samples.append({
                    "type": "textvqa", "image_obj": item['image'], 
                    "q": item['question'], "a": item['answers'][0]
                })
        except: pass

        # Shuffle & Limit
        if len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            
        print(f"✅ Final Dataset Ready: {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Image Loading
        try:
            if item['type'] == 'llava':
                img = cv2.imread(item['path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(item['image_obj'].convert("RGB"))

            # Simple Resize
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            pixel_values = processor(img, return_tensors="pt").pixel_values[0]
        except:
            pixel_values = torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE))

        # Text Formatting
        text = "<|im_start|>system\nYou are a helpful visual assistant.<|im_end|>\n"
        if item['type'] == 'llava':
            for turn in item['conv']:
                role = "user" if turn['from'] == 'human' else "assistant"
                text += f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n"
        else:
            text += f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n{item['a']}<|im_end|>\n"
            
        enc = tokenizer(text, truncation=True, max_length=config.MAX_LEN, return_tensors="pt")
        
        return {
            "pixel_values": pixel_values,
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0]
        }

def collate_fn(batch):
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
    
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
class CustomTrainer(Trainer):
    def place_model(self, model, training=True):
        return 

if __name__ == "__main__":
    model = QwenVQA()
    # Handle device placement manually if needed, or let Accelerate handle it
    if torch.cuda.is_available():
        model.cuda()
    
    train_dataset = HybridTrainDataset(config.LLAVA_JSON, config.COCO_TRAIN_DIR, max_samples=config.MAX_SAMPLES)
    
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        learning_rate=config.LR,
        fp16=True,
        logging_steps=10,
        remove_unused_columns=False
    )
    
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    trainer.train()
    
    # Save Final Models
    print("💾 Saving Model...")
    model.llm.save_pretrained(os.path.join(config.OUTPUT_DIR, "final_lora_model"))
    torch.save(model.connector.state_dict(), os.path.join(config.OUTPUT_DIR, "final_connector.pth"))
    print("✅ Saved!")