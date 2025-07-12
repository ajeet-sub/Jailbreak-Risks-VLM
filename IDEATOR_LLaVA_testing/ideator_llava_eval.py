import os, json, time, ast, logging, pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Pipeline
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# ————— Logging —————
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("debug_llava.log"), logging.StreamHandler()])

# ————— Load LLaVA —————
model_id = "llava-hf/llava-1.5-7b-hf"
print("Check =========)")
processor = LlavaProcessor.from_pretrained(model_id)
print("Check===========")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

# ————— Diffusion —————
pipe_sd = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
).to("cuda")

# ————— Helpers —————
def extract_json(s):
    try:
        data = ast.literal_eval(s[s.find("{"):s.rfind("}")+1].replace("\n",""))
        if "image prompt" in data and "text prompt" in data:
            return data
    except Exception:
        pass
    logging.error("Failed to extract JSON from attacker output.")
    return None

def query_llava(image, prompt, max_new_tokens=200):
    inputs = processor(prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)

# ————— Main —————
df = pd.read_csv("harmful_corpus/harmful_behaviors_custom.csv")
goals = df.iloc[:,1].tolist()[:1]  # only first example

os.makedirs("redteam_images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

for idx, goal in enumerate(goals):
    logging.info(f"[GOAL {idx}] {goal}")

    # Placeholder attacker (replace later)
    attacker_response = "{'image prompt': 'A photo of a locked safe in a dark room', 'text prompt': 'Write a fictional scenario...'}"
    attack = extract_json(attacker_response)
    if not attack: continue

    img_prompt, txt_prompt = attack["image prompt"], attack["text prompt"]
    
    img = pipe_sd(img_prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    img_path = f"redteam_images/{idx}.png"
    img.save(img_path)
    logging.info(f"Saved image → {img_path}")

    # Prepare prompt
    full_prompt = "<image>\nUSER: " + txt_prompt + "\nASSISTANT:"
    llava_output = query_llava(img, full_prompt)

    logging.info(f"LLaVA response:\n{llava_output}")
    pd.DataFrame([[idx,goal,img_prompt,txt_prompt,llava_output]], 
                 columns=["idx","goal","img","txt","response"]).to_csv(
        "outputs/model_ans_llava.csv", mode="a", index=False, header=not os.path.exists("outputs/model_ans_llava.csv")
    )
