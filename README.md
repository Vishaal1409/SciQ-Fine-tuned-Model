# SciQ Fine-tuned Model 🔬

A fine-tuned Large Language Model (LLM) for science question answering, built by fine-tuning Microsoft's **Phi-3 Mini (3.8B)** on the **SciQ dataset** using **LoRA** and **4-bit quantization** — running entirely on a local GPU.

---

## 🧠 What This Project Does

This project fine-tunes a 3.8 billion parameter LLM to accurately answer general science questions. Given a science question, the model produces a concise, correct answer based on what it learned from 11,679 science QA examples.

**Example:**
```
Question: What is the powerhouse of the cell?
Answer:   mitochondria

Question: What gas do plants absorb during photosynthesis?
Answer:   carbon dioxide (co2) gas from the air and water from the soil

Question: What force keeps planets in orbit around the sun?
Answer:   gravity force of attraction between planets and sun

Question: What is the chemical symbol for water?
Answer:   h2o

Question: What type of energy does the sun produce?
Answer:   light energy and heat energy (solar energy) from nuclear fusion reactions in its core
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Programming language |
| PyTorch 2.5.1 + CUDA | GPU-accelerated deep learning |
| Hugging Face Transformers | Loading and running the Phi-3 model |
| PEFT (LoRA) | Efficient fine-tuning — only 0.16% of parameters trained |
| TRL (SFTTrainer) | Supervised fine-tuning trainer |
| BitsAndBytes | 4-bit quantization to fit model in 6GB VRAM |
| SciQ Dataset | 11,679 science QA pairs for training |

---

## 💻 Hardware Used

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **OS:** Windows 11
- **Training time:** ~2 hours

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Vishaal1409/sciq-finetuned-model
cd sciq-finetuned-model
```

### 2. Create and activate conda environment
```bash
conda create -n llm-finetune python=3.11 -y
conda activate llm-finetune
```

### 3. Install dependencies
```bash
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 trl==0.11.4 peft accelerate bitsandbytes datasets
```

### 4. Prepare the dataset
```bash
python prepare_data.py
```

### 5. Fine-tune the model
```bash
python train.py
```
> ⚠️ Requires a CUDA-capable GPU with at least 6GB VRAM. Training takes ~2 hours on an RTX 4050.

### 6. Test the model
```bash
python test.py
```

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/Phi-3-mini-4k-instruct |
| Dataset | allenai/sciq (11,679 train examples) |
| Training Epochs | 2 |
| LoRA Rank (r) | 16 |
| Trainable Parameters | 6,291,456 (0.16% of total) |
| Batch Size | 2 |
| Learning Rate | 2e-4 |
| Quantization | 4-bit NF4 |
| Max Sequence Length | 256 tokens |

---

## 📚 What I Learned

This was my **first solo ML project** and I learned a huge amount throughout the process:

- **How LLMs work** — understanding that models are just very large collections of numbers (parameters) that can be adjusted through training
- **Fine-tuning vs training from scratch** — fine-tuning takes an existing powerful model and specializes it for a task, which is far more efficient than training from zero
- **LoRA (Low-Rank Adaptation)** — instead of updating all 3.8 billion parameters, LoRA adds small trainable layers so we only update 0.16% of the model — making fine-tuning possible on consumer hardware
- **4-bit Quantization** — compressing model weights from 32-bit to 4-bit numbers so a 3.8B parameter model fits in just 6GB of VRAM
- **Dataset formatting** — how raw data needs to be structured into a consistent prompt/response format for the model to learn from it
- **GPU environment setup** — setting up CUDA, managing Python environments, handling library version conflicts

---

## 🔮 Future Improvements

- Train for more epochs to reduce loss further
- Use a larger or domain-specific dataset for better accuracy
- Add a simple web UI using Gradio so anyone can chat with the model
- Try larger models (7B) with more VRAM

---

## 👤 Author

**Vishaal** — [@Vishaal1409](https://github.com/Vishaal1409)

*Built as a learning project to get hands-on experience with LLM fine-tuning, LoRA, and GPU-based training.*
