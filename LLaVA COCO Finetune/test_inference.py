
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO

BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CHECKPOINT_PATH = "./llava-coco-finetuned"  # or "./llava-coco-finetuned/checkpoint-epoch-1"

print("Loading base model and processor...")

processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

# Option 2: Load without quantization (if you have enough memory)
# base_model = LlavaForConditionalGeneration.from_pretrained(
#     BASE_MODEL_ID,
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )
# model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

# Option 3: Merge LoRA weights into base model (for faster inference)
# model = model.merge_and_unload()

model.eval()
print("Model loaded successfully!")

def generate_caption(image_url_or_path):
    """Generate caption for an image"""
    if image_url_or_path.startswith('http'):
        response = requests.get(image_url_or_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_url_or_path).convert('RGB')
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    if "ASSISTANT:" in generated_text:
        caption = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        caption = generated_text
    
    return caption

print("\nTesting inference...")
test_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
caption = generate_caption(test_url)
print(f"Generated caption: {caption}")