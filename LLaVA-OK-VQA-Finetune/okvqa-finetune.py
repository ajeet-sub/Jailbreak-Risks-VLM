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
DATASET_NAME = "HuggingFaceM4/OK-VQA"
OUTPUT_DIR = "./llava-okvqa-finetuned"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048
COCO_IMAGE_BASE = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
COCO_IMAGE_VAL_BASE = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

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

print("Loading OK-VQA dataset...")
dataset = load_dataset(DATASET_NAME)

def preprocess_single_example(example):
    """Preprocess a single OK-VQA example"""
    try:
        # Get image from COCO
        image_id = example['image_id']
        
        # Try train2014 first, then val2014 if that fails
        try:
            image_url = COCO_IMAGE_BASE.format(image_id)
            response = requests.get(image_url, timeout=5)
            if response.status_code != 200:
                image_url = COCO_IMAGE_VAL_BASE.format(image_id)
                response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            return None
        
        # Get question
        question = example['question']
        
        # Get answer - OK-VQA has multiple answers (list of dicts with 'answer' key)
        # Pick a random answer from the list
        if isinstance(example['answers'], list) and len(example['answers']) > 0:
            # Answers are in format: [{'answer': 'text', 'answer_confidence': ...}, ...]
            if isinstance(example['answers'][0], dict) and 'answer' in example['answers'][0]:
                answer = random.choice(example['answers'])['answer']
            else:
                # Fallback if format is different
                answer = random.choice(example['answers'])
        else:
            return None
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": answer}
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
        
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Mask image tokens
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        labels[labels == image_token_id] = -100
        
        # Mask instruction part - only train on assistant response
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
        print(f"Error processing example: {e}")
        return None

print("Debugging first example:")
first_example = dataset['train'][0]
print(f"Keys: {first_example.keys()}")
print(f"Image ID: {first_example['image_id']}")
print(f"Question: {first_example['question']}")
print(f"Answers type: {type(first_example['answers'])}")
print(f"Answers content: {first_example['answers'][:2]}")  # Print first 2 answers
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
# OK-VQA train split has ~9k examples
train_dataset = dataset['train'].select(range(min(5000, len(dataset['train']))))

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
    dataloader_num_workers=0,
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

# Test on a validation example if available, otherwise use train
test_split = 'val' if 'val' in dataset else 'train'
test_item = dataset[test_split][0]

try:
    test_image_url = COCO_IMAGE_BASE.format(test_item['image_id'])
    try:
        response = requests.get(test_image_url, timeout=5)
        if response.status_code != 200:
            test_image_url = COCO_IMAGE_VAL_BASE.format(test_item['image_id'])
            response = requests.get(test_image_url, timeout=5)
        test_image = Image.open(BytesIO(response.content)).convert('RGB')
    except:
        print("Could not load test image")
        test_image = None
    
    if test_image:
        test_question = test_item['question']
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": test_question}
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
        print(f"\nQuestion: {test_question}")
        print(f"Generated: {generated_text}")
        print(f"Ground truth answers: {[a['answer'] if isinstance(a, dict) else a for a in test_item['answers'][:3]]}")
    
except Exception as e:
    print(f"Inference test failed: {e}")

print("\nTraining complete!")