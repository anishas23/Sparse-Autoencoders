# qlora_tinyllama.py
# Finetunes a small chat LLM with QLoRA (4-bit) if bitsandbytes+CUDA are available,
# otherwise falls back to plain LoRA so it still runs on CPU/Windows.

import os, sys, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

print("🔧 Environment:", "CUDA" if torch.cuda.is_available() else "CPU")
device_map = "auto"

# ------------ Choose a small open chat model -------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ------------ Check bitsandbytes availability ------------
use_bnb = False
try:
    import bitsandbytes as bnb  # noqa: F401
    if torch.cuda.is_available():
        use_bnb = True
        print("✅ bitsandbytes found, will use 4-bit NF4 QLoRA.")
    else:
        print("ℹ️ bitsandbytes found but CUDA not available; running LoRA without 4-bit.")
except Exception as e:
    print("ℹ️ bitsandbytes not available; running LoRA without 4-bit.")

# ------------ Build quantization config if possible ------------
bnb_config = None
if use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

# ------------ Load tokenizer & base model ------------
print("📥 Loading base model:", BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if use_bnb:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model)
else:
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

# ------------ LoRA configuration ------------
# Target modules cover LLaMA/Mistral-style attention/MLP projections.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_config)
print("🧩 LoRA adapters injected.")

# ------------ Tiny toy instruction dataset ------------
pairs = [
    {"instruction":"Translate to Hindi: 'Hello, how are you?'", "output":"नमस्ते, आप कैसे हैं?"},
    {"instruction":"Explain QLoRA in one sentence.", "output":"QLoRA 4-बिट क्वांटाइज़ेशन और LoRA अडैप्टर का इस्तेमाल करके बड़े मॉडल को कम मेमोरी में फाइन-ट्यून करने की विधि है."},
    {"instruction":"Summarize: 'Apples are fruits rich in fiber.'", "output":"सेब फाइबर से भरपूर फल हैं."},
    {"instruction":"Write a short advice for studying effectively.", "output":"दिन का छोटा लक्ष्य बनाएं, बाधाएँ हटाएँ, दोहराएँ और ब्रेक लें."},
]
def format_example(ex):
    # Simple instruction-following pattern
    return f"<s>Instruction:\n{ex['instruction']}\n\nResponse:\n{ex['output']}</s>"

raw_ds = Dataset.from_list(pairs)
text_ds = raw_ds.map(lambda ex: {"text": format_example(ex)})

def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    out["labels"] = out["input_ids"].copy()
    return out

tok_ds = text_ds.map(tokenize, batched=True, remove_columns=text_ds.column_names)

# ------------ Training setup ------------
per_device_bs = 2 if torch.cuda.is_available() else 1
grad_accum = 8 if torch.cuda.is_available() else 16
optim_name = "paged_adamw_32bit" if use_bnb else "adamw_torch"

args = TrainingArguments(
    output_dir="qlora-out",
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),          # enables bf16 on supporting GPUs
    gradient_checkpointing=True,
    optim=optim_name,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    eval_dataset=None,
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")

# ------------ Save LoRA adapters ------------
ADAPTER_DIR = "qlora_tinyllama_lora"
os.makedirs(ADAPTER_DIR, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"💾 Adapters saved to: {ADAPTER_DIR}")

# ------------ Inference using the trained adapters ------------
print("🔁 Running a sample generation...")

# Reload base + adapters cleanly (as you would in a separate script)
if use_bnb:
    gen_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map=device_map,
    )
else:
    gen_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map=device_map,
    )
gen_model = PeftModel.from_pretrained(gen_model, ADAPTER_DIR)

def chat(prompt: str, max_new_tokens=80):
    text = f"<s>Instruction:\n{prompt}\n\nResponse:\n"
    inputs = tokenizer(text, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

tests = [
    "Write a haiku about monsoons in India.",
    "Translate to Hindi: 'Good morning, friend!'",
    "Explain QLoRA very simply.",
]

outputs = []
for t in tests:
    ans = chat(t)
    print("\n---")
    print("Instruction:", t)
    print("Model:", ans)
    outputs.append(f"Instruction: {t}\nModel: {ans}\n")

with open("qlora_sample_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(outputs))
print("📝 Saved generations to qlora_sample_output.txt")
print("🎉 Done.")
