# diff_full.py
import torch, hashlib, re
from collections import Counter
from transformers import LlavaForConditionalGeneration
from peft import PeftModel

BASE = "llava-hf/llava-1.5-7b-hf"
ADAPT = "./llava-coco-finetuned"

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

base = LlavaForConditionalGeneration.from_pretrained(BASE, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
pre = {k: hashlib.md5(v.float().cpu().numpy().tobytes()).hexdigest() for k,v in base.state_dict().items()}

merged = PeftModel.from_pretrained(base, ADAPT).merge_and_unload()
post = {k: hashlib.md5(v.float().cpu().numpy().tobytes()).hexdigest() for k,v in merged.state_dict().items()}

changed = [k for k in post if pre.get(k) != post[k]]

bucket = []
for k in changed:
    if k.startswith("model.vision_tower."):
        bucket.append("vision_tower")
    elif k.startswith("model.language_model."):
        if ".self_attn." in k: bucket.append("LM_attn")
        elif ".mlp." in k:    bucket.append("LM_mlp")
        else:                  bucket.append("LM_other")
    elif "mm_projector" in k:
        bucket.append("mm_projector")
    else:
        bucket.append("other")
cnt = Counter(bucket)

print(f"Total params: {len(post):,}")
print(f"Changed params: {len(changed):,}")
for b,c in cnt.most_common():
    print(f"{b:15s}: {c:,}")
