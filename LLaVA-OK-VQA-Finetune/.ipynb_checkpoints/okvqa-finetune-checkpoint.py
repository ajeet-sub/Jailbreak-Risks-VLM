import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import random
import os
import numpy as np

# ============================================
# Weights & Biases Integration
# ============================================
import wandb

# Initialize W&B - set your project name and optional run name
WANDB_PROJECT = "llava-okvqa-finetune"
WANDB_RUN_NAME = "llava-1.5-7b-lora-okvqa"

wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        # Model config
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "dataset": "HuggingFaceM4/OK-VQA",
        
        # LoRA config
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Training config
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 2 * 8,  # batch_size * grad_accum
        "num_epochs": 5,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler": "cosine",
        "max_length": 2048,
        "quantization": "4bit-nf4",
    }
)

# ============================================
# Config
# ============================================
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME = "HuggingFaceM4/OK-VQA"
OUTPUT_DIR = "./llava-okvqa-finetuned"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 2048
COCO_IMAGE_BASE = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
COCO_IMAGE_VAL_BASE = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================
# Custom W&B Callback for Weight Tracking
# ============================================
class WandBWeightTrackingCallback(TrainerCallback):
    """
    Custom callback to track LoRA weight statistics and movement during training.
    Logs weight norms, gradient norms, and weight changes to W&B.
    """
    
    def __init__(self, model, log_freq=50):
        self.model = model
        self.log_freq = log_freq
        self.initial_weights = {}
        self.previous_weights = {}
        self._store_initial_weights()
    
    def _get_lora_params(self):
        """Get all LoRA parameters from the model."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params[name] = param
        return lora_params
    
    def _store_initial_weights(self):
        """Store initial weights for tracking total movement."""
        for name, param in self._get_lora_params().items():
            self.initial_weights[name] = param.detach().clone().cpu()
            self.previous_weights[name] = param.detach().clone().cpu()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log weight statistics at regular intervals."""
        if state.global_step % self.log_freq != 0:
            return
        
        lora_params = self._get_lora_params()
        
        # Collect statistics
        weight_stats = {
            'lora_A': {'norms': [], 'grad_norms': [], 'changes': [], 'total_changes': []},
            'lora_B': {'norms': [], 'grad_norms': [], 'changes': [], 'total_changes': []},
        }
        
        all_weight_norms = []
        all_grad_norms = []
        all_step_changes = []
        all_total_changes = []
        
        for name, param in lora_params.items():
            # Determine if this is lora_A or lora_B
            lora_type = 'lora_A' if 'lora_A' in name else 'lora_B' if 'lora_B' in name else None
            
            # Weight norm
            weight_norm = param.detach().norm().item()
            all_weight_norms.append(weight_norm)
            
            # Gradient norm (if available)
            grad_norm = 0.0
            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
            all_grad_norms.append(grad_norm)
            
            # Step-wise weight change
            current_weight = param.detach().cpu()
            if name in self.previous_weights:
                step_change = (current_weight - self.previous_weights[name]).norm().item()
                all_step_changes.append(step_change)
            
            # Total weight change from initialization
            if name in self.initial_weights:
                total_change = (current_weight - self.initial_weights[name]).norm().item()
                all_total_changes.append(total_change)
            
            # Update previous weights
            self.previous_weights[name] = current_weight.clone()
            
            # Collect by LoRA type
            if lora_type:
                weight_stats[lora_type]['norms'].append(weight_norm)
                weight_stats[lora_type]['grad_norms'].append(grad_norm)
                if name in self.previous_weights:
                    weight_stats[lora_type]['changes'].append(step_change)
                if name in self.initial_weights:
                    weight_stats[lora_type]['total_changes'].append(total_change)
        
        # Log to W&B
        log_dict = {
            "weights/mean_weight_norm": np.mean(all_weight_norms) if all_weight_norms else 0,
            "weights/max_weight_norm": np.max(all_weight_norms) if all_weight_norms else 0,
            "weights/mean_grad_norm": np.mean(all_grad_norms) if all_grad_norms else 0,
            "weights/max_grad_norm": np.max(all_grad_norms) if all_grad_norms else 0,
            "weights/mean_step_change": np.mean(all_step_changes) if all_step_changes else 0,
            "weights/mean_total_change": np.mean(all_total_changes) if all_total_changes else 0,
            "weights/max_total_change": np.max(all_total_changes) if all_total_changes else 0,
        }
        
        # Add LoRA A/B specific stats
        for lora_type in ['lora_A', 'lora_B']:
            stats = weight_stats[lora_type]
            if stats['norms']:
                log_dict[f"weights/{lora_type}_mean_norm"] = np.mean(stats['norms'])
                log_dict[f"weights/{lora_type}_mean_grad_norm"] = np.mean(stats['grad_norms'])
            if stats['total_changes']:
                log_dict[f"weights/{lora_type}_mean_total_change"] = np.mean(stats['total_changes'])
        
        wandb.log(log_dict, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch-level weight statistics."""
        lora_params = self._get_lora_params()
        
        # Calculate per-layer total movement
        layer_movements = {}
        for name, param in lora_params.items():
            if name in self.initial_weights:
                current_weight = param.detach().cpu()
                movement = (current_weight - self.initial_weights[name]).norm().item()
                
                # Extract layer identifier
                layer_id = name.split('.')[0:4]  # Adjust based on model structure
                layer_key = '.'.join(layer_id)
                
                if layer_key not in layer_movements:
                    layer_movements[layer_key] = []
                layer_movements[layer_key].append(movement)
        
        # Log per-layer movements as a bar chart
        layer_avg_movements = {k: np.mean(v) for k, v in layer_movements.items()}
        
        # Create a table for layer movements
        table_data = [[layer, movement] for layer, movement in sorted(layer_avg_movements.items())]
        movement_table = wandb.Table(data=table_data, columns=["Layer", "Weight Movement"])
        
        wandb.log({
            f"epoch_{int(state.epoch)}/layer_weight_movements": wandb.plot.bar(
                movement_table, "Layer", "Weight Movement",
                title=f"Weight Movement by Layer (Epoch {int(state.epoch)})"
            )
        }, step=state.global_step)


class WandBGradientHistogramCallback(TrainerCallback):
    """
    Callback to log gradient histograms periodically.
    """
    
    def __init__(self, model, log_freq=100):
        self.model = model
        self.log_freq = log_freq
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq != 0:
            return
        
        # Collect gradient values
        all_grads = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad and param.grad is not None:
                all_grads.extend(param.grad.detach().cpu().flatten().numpy())
        
        if all_grads:
            wandb.log({
                "gradients/histogram": wandb.Histogram(all_grads),
            }, step=state.global_step)


# ============================================
# Model Loading
# ============================================
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

# Improved LoRA config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Log model architecture to W&B
wandb.watch(model, log="all", log_freq=100)

print("Loading OK-VQA dataset...")
dataset = load_dataset(DATASET_NAME)


# ============================================
# Data Preprocessing
# ============================================
def preprocess_single_example(example):
    """Preprocess a single OK-VQA example"""
    try:
        image_id = example['image_id']
        
        try:
            image_url = COCO_IMAGE_BASE.format(image_id)
            response = requests.get(image_url, timeout=5)
            if response.status_code != 200:
                image_url = COCO_IMAGE_VAL_BASE.format(image_id)
                response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            return None
        
        question = example['question']
        
        if isinstance(example['answers'], list) and len(example['answers']) > 0:
            if isinstance(example['answers'][0], dict) and 'answer' in example['answers'][0]:
                answer = random.choice(example['answers'])['answer']
            else:
                answer = random.choice(example['answers'])
        else:
            return None
        
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
        print(f"Error processing example: {e}")
        return None


print("Debugging first example:")
first_example = dataset['train'][0]
print(f"Keys: {first_example.keys()}")
print(f"Image ID: {first_example['image_id']}")
print(f"Question: {first_example['question']}")
print(f"Answers type: {type(first_example['answers'])}")
print(f"Answers content: {first_example['answers'][:2]}")
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
train_dataset = dataset['train']  # Use full dataset

processed_data = []

for i, example in enumerate(train_dataset):
    processed = preprocess_single_example(example)
    if processed is not None:
        processed_data.append(processed)
    
    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{len(train_dataset)} examples, got {len(processed_data)} valid examples")

print(f"Total valid examples: {len(processed_data)}")

# Log dataset statistics to W&B
wandb.log({
    "dataset/total_examples": len(train_dataset),
    "dataset/valid_examples": len(processed_data),
    "dataset/success_rate": len(processed_data) / len(train_dataset) * 100
})


# ============================================
# Training Configuration
# ============================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.03,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    
    # W&B integration
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
)

# Create custom callbacks
weight_tracking_callback = WandBWeightTrackingCallback(model, log_freq=50)
gradient_histogram_callback = WandBGradientHistogramCallback(model, log_freq=100)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data,
    data_collator=collate_fn,
    callbacks=[weight_tracking_callback, gradient_histogram_callback],
)


# ============================================
# Training
# ============================================
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Log final model artifact to W&B
artifact = wandb.Artifact(
    name="llava-okvqa-lora",
    type="model",
    description="LoRA fine-tuned LLaVA on OK-VQA"
)
artifact.add_dir(OUTPUT_DIR)
wandb.log_artifact(artifact)


# ============================================
# Inference Test
# ============================================
print("\nTesting inference...")
model.eval()

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
        ground_truth = [a['answer'] if isinstance(a, dict) else a for a in test_item['answers'][:3]]
        
        print(f"\nQuestion: {test_question}")
        print(f"Generated: {generated_text}")
        print(f"Ground truth answers: {ground_truth}")
        
        # Log inference example to W&B
        wandb.log({
            "inference/test_question": test_question,
            "inference/generated_answer": generated_text,
            "inference/ground_truth": str(ground_truth),
            "inference/test_image": wandb.Image(test_image, caption=test_question)
        })
    
except Exception as e:
    print(f"Inference test failed: {e}")

# Finish W&B run
wandb.finish()

print("\nTraining complete!")