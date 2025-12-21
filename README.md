# 👁️ Qwen2.5-VQA: Multi-Dataset Vision-Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-Trained-success)

A specialized Multimodal Large Language Model (MLLM) built by fusing a **Vision Transformer (ViT)** with **Qwen2.5-3B-Instruct**. Unlike standard implementations, this model features a custom trainable connector and was fine-tuned on a **diverse triad of datasets** to handle general visual conversations, text reading (OCR), and complex visual reasoning.

## 🌟 Key Functionalities

This model isn't just a generic chatbot; it has been trained on **3 distinct datasets** to master specific visual skills:

| Dataset            | Skill Acquired                     | Functionality                                                                                                                            |
| :----------------- | :--------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **LLaVA Instruct** | 🗣️ **Visual Conversation**         | Capable of open-ended chat, describing scenes in detail, and following complex visual instructions.                                      |
| **TextVQA**        | 📖 **OCR / Reading**               | Can detect and read text inside images (posters, books, signs) and answer questions based on that text.                                  |
| **VizWiz**         | 🔍 **Fine Detail & Accessibility** | Trained to answer questions often asked by visually impaired users, focusing on practical details (labels, colors, identifying objects). |

---

## 🏗️ Architecture

The model connects a frozen vision encoder to a generative text decoder through a trainable projection layer.

````mermaid
graph LR
    A[Image Input] -->|Patching| B[ViT Encoder]
    B -->|Features 768 dim| C[Connector MLP]
    C -->|Projected Embeds 2048 dim| D[Qwen Embedding Space]
    E[Text Prompts] -->|Tokenization| D
    D -->|Concatenation| F[Qwen2.5 LLM]
    F -->|Generation| G[Answer]
````
1.  **Vision Encoder:** `google/vit-base-patch32-384` (Frozen). Extracts high-level visual features.
2.  **Projection Layer (The "Connector"):** A custom 2-layer Multi-Layer Perceptron (Linear -> GELU -> Linear). It maps the visual vector space ($R^{768}$) to the LLM's token vector space ($R^{2048}$).
3.  **LLM Backbone:** `Qwen/Qwen2.5-3B-Instruct`. We inject **LoRA adapters** into the attention mechanism (`q_proj`, `v_proj`, `k_proj`, `o_proj`) to fine-tune the model's reasoning capabilities without retraining the full weights.

---

## 📦 Installation

### Prerequisites
* Python 3.10+
* NVIDIA GPU with at least 12GB VRAM (for training) or 8GB (for inference).
* CUDA Toolkit installed.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ishreyanshkumar/VQA.git](https://github.com/ishreyanshkumar/VQA.git)
    cd VQA

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install streamlit  # Required for the Web UI

---

## 📥 Pre-trained Weights

To run inference without training from scratch, download the trained weights and place them in the `output/` directory.

| Component | Filename | Description | Download Link |
| :--- | :--- | :--- | :--- |
| **Visual Connector** | `final_connector.pth` | Maps ViT features to Qwen space | [🔗 https://drive.google.com/file/d/16wqNg8PQOL05ET25NKpxCzsnA8LLcscd/view?usp=sharing](#) |
| **LoRA Adapters** | `final_lora_model/` | Fine-tuned Qwen attention layers | [🔗 https://drive.google.com/file/d/16wqNg8PQOL05ET25NKpxCzsnA8LLcscd/view?usp=sharing](#) |

**Directory setup after downloading:**
```text
Qwen2.5-VQA/
└── output/
    ├── final_connector.pth
    └── final_lora_model/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

---

## 📂 Project Structure

```text
Qwen2.5-VQA/
├── train.py                # Training pipeline
├── app.py                  # Streamlit Web UI for inference
├── requirements.txt        # Python dependencies
├── output/                 # (Create this) Place downloaded weights here
└── data/                   # Dataset folder (only needed for training)
    ├── llava_instruct_150k.json
    └── coco2017/
```

---

## 🏋️‍♀️ Training

To train the connector and LoRA adapters on the mixed dataset:

```bash
python train.py
```

### Hyperparameters
The default configuration (found in `train.py`) is optimized for a T4 or A10 GPU:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Image Size** | 384x384 | Resized via OpenCV with padding (no aspect ratio distortion) |
| **Batch Size** | 4 | Small batch size to fit in VRAM |
| **Grad Accum** | 16 | Effective batch size = 64 |
| **LoRA Rank** | 64 | High rank for better feature adaptation |
| **Precision** | FP16 | Mixed precision training |
| **Epochs** | 1 | Sufficient for connector convergence |

**Output:**
Training will produce two critical files in the `output/` directory:
1.  `final_connector.pth`: The weights for the visual projection layer.
2.  `final_lora_model/`: The directory containing the LoRA adapter weights.

---

## 🤖 Inference (Streamlit Web UI)

We provide a polished web interface using Streamlit. This supports **image uploads**, **URL inputs**, and **persistent chat history**.

### Run the App
```bash
streamlit run app.py
```

### Features
* **Sidebar Upload:** Easily switch between local images or image URLs.
* **Memory:** The bot remembers previous questions in the conversation (e.g., "What is in the image?" -> "What color is it?").
* **Efficient Caching:** The 3B model loads only once when the server starts, making interaction snappy.

---

## 🔧 Technical Deep Dive

### 1. The Tokenization Strategy
We use the ChatML format native to Qwen. The visual embeddings are prepended to the text embeddings.
```text
<|im_start|>system
You are a helpful visual assistant.
<|im_end|>
<|im_start|>user
[IMAGE EMBEDDINGS] Describe this image.
<|im_end|>
<|im_start|>assistant
```

### 2. Smart Resizing
Standard resizing squashes images, destroying spatial relationships. We use a **Canvas Padding** approach:
1.  Scale the image so the longest side matches 384px.
2.  Paste the scaled image onto a black 384x384 canvas.
3.  This preserves aspect ratio, crucial for OCR and object detection tasks.

---
## 🚀 Future Improvements

This project represents a solid baseline for a custom VQA architecture. However, several enhancements can be implemented to push performance closer to State-of-the-Art (SOTA) levels.

### 1. Vision Encoder Upgrade
* **Current:** `ViT-Base-Patch32-384` (optimized primarily for classification).
* **Proposed:** Switch to **SigLIP-SO400M** or **CLIP-Large**.
* **Why:** These models are specifically trained for image-text alignment rather than just classification, offering significantly better semantic understanding of complex scenes (e.g., distinguishing "a dog on a beach" vs. just "dog features").

### 2. Advanced LoRA Tuning
* **Current:** LoRA target modules are limited to attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
* **Proposed:** Target **MLP layers** as well (`gate_proj`, `up_proj`, `down_proj`).
* **Why:** Research indicates that MLP layers in LLMs store a vast amount of factual knowledge. Fine-tuning them improves the model's reasoning capabilities and reduces hallucinations.

### 3. Dynamic Resolution (AnyRes) Support
* **Current:** All images are resized to a fixed square ($384 \times 384$), which causes distortion for tall or wide images.
* **Proposed:** Implement a dynamic patching strategy (similar to LLaVA-NeXT or Qwen-VL).
* **Why:** This involves cropping high-resolution images into multiple grids (e.g., 4 sub-patches) and feeding them alongside a global context view. This allows the model to "read" small text and see fine details without losing the overall context.

### 4. Two-Stage Training Strategy
* **Current:** Single-stage training where the connector and LoRA adapters are trained simultaneously.
* **Proposed:**
    * **Stage 1 (Pre-alignment):** Freeze the LLM and Vision Encoder; train *only* the Connector on large-scale caption pairs.
    * **Stage 2 (Fine-tuning):** Unfreeze the LoRA adapters and train on high-quality instruction data.
* **Why:** This "warm-up" phase ensures the visual features are properly aligned with the language space before the LLM tries to learn complex instructions, leading to faster convergence and better stability.

### 5. Deployment Optimization
* **Current:** Standard PyTorch inference.
* **Proposed:**
    * **Quantization:** Implement 4-bit (QLoRA) or 8-bit inference for lower memory usage.
    * **vLLM / TGI:** Serve the model using high-performance inference engines like vLLM for faster token generation and concurrent user support.
