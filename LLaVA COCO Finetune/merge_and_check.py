# merge_and_check.py
from transformers import LlavaForConditionalGeneration, AutoTokenizer
from peft import PeftModel
import torch, os, json, hashlib

BASE = "llava-hf/llava-1.5-7b-hf"         # replace if #1 says otherwise
ADAPT = "./llava-coco-finetuned"
OUT   = "./llava-merged-finetune"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Load base
base = LlavaForConditionalGeneration.from_pretrained(BASE, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
# Snapshot a few param hashes before merge
keys = [k for k,_ in list(base.named_parameters())[:12]]
pre = {k: hashlib.md5(base.state_dict()[k].float().cpu().numpy().tobytes()).hexdigest() for k in keys}

# Apply LoRA
model = PeftModel.from_pretrained(base, ADAPT)
model = model.merge_and_unload()   # merge into base

# Compare hashes again
post = {k: hashlib.md5(model.state_dict()[k].float().cpu().numpy().tobytes()).hexdigest() for k in keys}
diffs = [k for k in keys if pre[k] != post[k]]
print("Changed params (sample):", diffs[:8] or "(none!)")

# Save merged + tokenizer
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
# Prefer tokenizer from ADAPT if present, else BASE
try:
    tok = AutoTokenizer.from_pretrained(ADAPT, use_fast=True, trust_remote_code=True)
except Exception:
    tok = AutoTokenizer.from_pretrained(BASE,  use_fast=True, trust_remote_code=True)
tok.save_pretrained(OUT)
print("Merged checkpoint saved to", OUT)