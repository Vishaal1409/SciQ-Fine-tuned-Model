import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

# ── 1. Load the dataset
dataset = load_from_disk("sciq_formatted")
print("Dataset loaded!")

# ── 2. Set up 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 3. Load the model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded!")

# ── 4. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 5. Set up the trainer
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    warmup_steps=100,
    max_seq_length=256,
    dataset_text_field="text",
)

print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
)

# ── 6. Start training!
print("Trainer ready! Starting training...")
try:
    trainer.train()
    print("\nTraining complete!")
    # ── 7. Save the fine-tuned model
    model.save_pretrained("./sciq-finetuned")
    tokenizer.save_pretrained("./sciq-finetuned")
    print("Model saved to ./sciq-finetuned")
except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()