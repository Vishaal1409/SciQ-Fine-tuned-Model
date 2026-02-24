import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── 1. Same 4-bit quantization as training so model fits in VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_name = "microsoft/Phi-3-mini-4k-instruct"
print("Loading your fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ── 2. Load base model with quantization this time
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ── 3. Load your fine-tuned LoRA weights on top
model = PeftModel.from_pretrained(base_model, "./sciq-finetuned")
model.eval()
print("Model ready!\n")

# ── 4. Function to ask the model a question
def ask(question):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,        # Shorter answers
            temperature=0.1,          # Much more focused/deterministic
            do_sample=False,          # Greedy decoding — picks the most likely token every time
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Answer:")[-1].strip()
    # Stop at the first newline or question mark to avoid rambling
    answer = answer.split("\n")[0].split("?")[0]
    return answer

# ── 5. Test with science questions!
questions = [
    "What force keeps planets in orbit around the sun?",
    "What is the powerhouse of the cell?",
    "What gas do plants absorb during photosynthesis?",
    "What is the chemical symbol for water?",
    "What type of energy does the sun produce?",
]

for q in questions:
    print(f"Question: {q}")
    print(f"Answer:   {ask(q)}")
    print()