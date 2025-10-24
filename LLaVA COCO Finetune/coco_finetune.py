import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import random
import os

# Config
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME = "yerevann/coco-karpathy"
OUTPUT_DIR = "./llava-coco-finetuned"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading COCO dataset...")
dataset = load_dataset(DATASET_NAME)

def preprocess_single_example(example):
    """Preprocess a single example"""
    try:
        response = requests.get(example['url'], timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        if isinstance(example['sentences'], dict) and 'raw' in example['sentences']:
            captions = example['sentences']['raw']
            if len(captions) == 0:
                return None
            caption = random.choice(captions)
        elif isinstance(example['sentences'], list) and len(example['sentences']) > 0:
            caption = random.choice(example['sentences'])
        else:
            return None
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=False)
        
        # Process
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        labels = inputs["input_ids"].clone()
        
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        labels[labels == image_token_id] = -100
        
        assistant_start = text.find("ASSISTANT:")
        if assistant_start != -1:
            prefix_text = text[:assistant_start + len("ASSISTANT:")]
            prefix_tokens = processor.tokenizer(prefix_text, add_special_tokens=False)['input_ids']
            
            labels[0, :len(prefix_tokens)] = -100
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }
        
    except Exception as e:
        return None

print("Debugging first example:")
first_example = dataset['train'][0]
print(f"Keys: {first_example.keys()}")
print(f"Sentences type: {type(first_example['sentences'])}")
print(f"Sentences content: {first_example['sentences']}")
print()

def collate_fn(examples):
    examples = [ex for ex in examples if ex is not None]
    if len(examples) == 0:
        return None
    
    pixel_values = torch.stack([ex['pixel_values'] for ex in examples])
    
    input_ids = [ex['input_ids'] for ex in examples]
    labels = [ex['labels'] for ex in examples]
    
    max_len = max(ids.shape[0] for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for ids, lbls in zip(input_ids, labels):
        seq_len = ids.shape[0]
        pad_len = max_len - seq_len
        
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), processor.tokenizer.pad_token_id)])
        )
        padded_labels.append(
            torch.cat([lbls, torch.full((pad_len,), -100)])
        )
        attention_masks.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )
    
    return {
        'pixel_values': pixel_values,
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(padded_labels)
    }

print("Preprocessing dataset...")
train_dataset = dataset['train'].select(range(10000))  # Limit for faster training

processed_data = []

for i, example in enumerate(train_dataset):
    processed = preprocess_single_example(example)
    if processed is not None:
        processed_data.append(processed)
    
    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{len(train_dataset)} examples, got {len(processed_data)} valid examples")

print(f"Total valid examples: {len(processed_data)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,  # Avoid multiprocessing issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\nTesting inference...")
model.eval()

test_item = dataset['train'][0]
try:
    test_image = Image.open(BytesIO(requests.get(test_item['url']).content)).convert('RGB')
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]
    }]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")
    
except Exception as e:
    print(f"Inference test failed: {e}")

print("\nTraining complete!")