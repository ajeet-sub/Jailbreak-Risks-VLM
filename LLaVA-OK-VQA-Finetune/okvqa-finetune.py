import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
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
        "train_dataset": "Multimodal-Fatima/OK-VQA_train",
        "val_dataset": "lmms-lab/OK-VQA",
        
        # ============================================
        # LoRA config
        # ============================================
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        
        # ============================================
        # Training config
        # ============================================
        "batch_size": 4,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": 4 * 32,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler": "cosine",
        "max_length": 2048,
        "quantization": "none",
    }
)

# ============================================
# Config
# ============================================
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TRAIN_DATASET_NAME = "Multimodal-Fatima/OK-VQA_train"
VAL_DATASET_NAME = "lmms-lab/OK-VQA"
OUTPUT_DIR = "./llava-okvqa-finetuned"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 2048
WARMUP_RATIO = 0.03

# ============================================
# Val2014 split config
# ============================================
# OK-VQA has no separate test set. val2014 is the
# only eval split available. To avoid double-dipping
# (using the same data for checkpoint selection AND
# final evaluation), we split val2014 into:
#   60% for eval_loss during training (checkpoint selection)
#   40% held out for post-training evaluation
# ============================================
VAL_EVAL_RATIO = 0.6  # 60% of val2014 for training eval
VAL_TEST_RATIO = 0.4  # 40% of val2014 held out for final eval
VAL_SPLIT_SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================
# Custom W&B Callback for Weight Tracking
# ============================================
class WandBWeightTrackingCallback(TrainerCallback):
    """
    Custom callback to track LoRA weight statistics and movement during training.
    """
    
    def __init__(self, model, log_freq=50):
        self.model = model
        self.log_freq = log_freq
        self.initial_weights = {}
        self.previous_weights = {}
        self.weight_history = []
        self._store_initial_weights()
    
    def _get_lora_params(self):
        """Get all LoRA parameters from the model."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params[name] = param
        return lora_params
    
    def _get_module_type(self, name):
        """Extract the module type from parameter name."""
        for module in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            if module in name:
                return module
        return 'other'
    
    def _get_layer_idx(self, name):
        """Extract layer index from parameter name."""
        import re
        match = re.search(r'layers\.(\d+)\.', name)
        return int(match.group(1)) if match else -1
    
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
        
        # Per-module type stats
        module_stats = {mod: {'norms': [], 'grad_norms': [], 'total_changes': []} 
                       for mod in ['q_proj', 'v_proj']}
        
        all_weight_norms = []
        all_grad_norms = []
        all_step_changes = []
        all_total_changes = []
        
        for name, param in lora_params.items():
            lora_type = 'lora_A' if 'lora_A' in name else 'lora_B' if 'lora_B' in name else None
            module_type = self._get_module_type(name)
            
            weight_norm = param.detach().norm().item()
            all_weight_norms.append(weight_norm)
            
            grad_norm = 0.0
            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
            all_grad_norms.append(grad_norm)
            
            current_weight = param.detach().cpu()
            step_change = 0.0
            if name in self.previous_weights:
                step_change = (current_weight - self.previous_weights[name]).norm().item()
                all_step_changes.append(step_change)
            
            total_change = 0.0
            if name in self.initial_weights:
                total_change = (current_weight - self.initial_weights[name]).norm().item()
                all_total_changes.append(total_change)
            
            self.previous_weights[name] = current_weight.clone()
            
            if lora_type:
                weight_stats[lora_type]['norms'].append(weight_norm)
                weight_stats[lora_type]['grad_norms'].append(grad_norm)
                weight_stats[lora_type]['changes'].append(step_change)
                weight_stats[lora_type]['total_changes'].append(total_change)
            
            if module_type in module_stats:
                module_stats[module_type]['norms'].append(weight_norm)
                module_stats[module_type]['grad_norms'].append(grad_norm)
                module_stats[module_type]['total_changes'].append(total_change)
        
        log_dict = {
            "weights/mean_weight_norm": np.mean(all_weight_norms) if all_weight_norms else 0,
            "weights/max_weight_norm": np.max(all_weight_norms) if all_weight_norms else 0,
            "gradients/mean_grad_norm": np.mean(all_grad_norms) if all_grad_norms else 0,
            "weight_delta/mean_step_change": np.mean(all_step_changes) if all_step_changes else 0,
            "weight_movement/mean_total_change": np.mean(all_total_changes) if all_total_changes else 0,
            "weight_movement/sum_total_change": np.sum(all_total_changes) if all_total_changes else 0,
        }
        
        for lora_type in ['lora_A', 'lora_B']:
            stats = weight_stats[lora_type]
            if stats['norms']:
                log_dict[f"lora_weights/{lora_type}_mean_norm"] = np.mean(stats['norms'])
                log_dict[f"lora_gradients/{lora_type}_mean_grad_norm"] = np.mean(stats['grad_norms'])
            if stats['total_changes']:
                log_dict[f"lora_movement/{lora_type}_mean_total_change"] = np.mean(stats['total_changes'])
        
        for module_type, stats in module_stats.items():
            if stats['norms']:
                log_dict[f"module_weights/{module_type}_mean_norm"] = np.mean(stats['norms'])
            if stats['total_changes']:
                log_dict[f"module_movement/{module_type}_total_change"] = np.mean(stats['total_changes'])
        
        wandb.log(log_dict, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch-level weight statistics."""
        lora_params = self._get_lora_params()
        
        layer_movements = {}
        module_movements = {mod: [] for mod in ['q_proj', 'v_proj']}
        
        for name, param in lora_params.items():
            if name in self.initial_weights:
                current_weight = param.detach().cpu()
                movement = (current_weight - self.initial_weights[name]).norm().item()
                
                layer_idx = self._get_layer_idx(name)
                if layer_idx >= 0:
                    if layer_idx not in layer_movements:
                        layer_movements[layer_idx] = []
                    layer_movements[layer_idx].append(movement)
                
                module_type = self._get_module_type(name)
                if module_type in module_movements:
                    module_movements[module_type].append(movement)
        
        layer_avg_movements = {f"layer_{k}": np.mean(v) for k, v in sorted(layer_movements.items())}
        layer_table_data = [[layer, movement] for layer, movement in layer_avg_movements.items()]
        layer_table = wandb.Table(data=layer_table_data, columns=["Layer", "Weight Movement"])
        
        module_avg_movements = {k: np.mean(v) if v else 0 for k, v in module_movements.items()}
        module_table_data = [[mod, movement] for mod, movement in module_avg_movements.items()]
        module_table = wandb.Table(data=module_table_data, columns=["Module", "Weight Movement"])
        
        wandb.log({
            f"epoch_{int(state.epoch)}/layer_weight_movements": wandb.plot.bar(
                layer_table, "Layer", "Weight Movement",
                title=f"Weight Movement by Layer (Epoch {int(state.epoch)})"
            ),
            f"epoch_{int(state.epoch)}/module_weight_movements": wandb.plot.bar(
                module_table, "Module", "Weight Movement",
                title=f"Weight Movement by Module Type (Epoch {int(state.epoch)})"
            ),
            f"epoch_summary/epoch_{int(state.epoch)}_total_movement": sum(
                np.mean(v) for v in layer_movements.values()
            ),
        }, step=state.global_step)


class WandBGradientHistogramCallback(TrainerCallback):
    """Callback to log gradient histograms periodically."""
    
    def __init__(self, model, log_freq=100):
        self.model = model
        self.log_freq = log_freq
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq != 0:
            return
        
        all_grads = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad and param.grad is not None:
                all_grads.extend(param.grad.detach().cpu().flatten().numpy())
        
        if all_grads:
            wandb.log({
                "gradients/histogram": wandb.Histogram(all_grads),
            }, step=state.global_step)


# ============================================
# Lazy Dataset Wrapper
# ============================================
class VQADataset(Dataset):
    """Lazy dataset wrapper that preprocesses examples on-the-fly."""
    
    def __init__(self, hf_dataset, processor, max_length=2048, is_eval=False):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.is_eval = is_eval
        
        # Build index of valid examples (those with images and answers)
        self.valid_indices = []
        for i in range(len(hf_dataset)):
            example = hf_dataset[i]
            answers = example.get('answers', [])
            if example.get('image') is not None and answers and len(answers) > 0:
                self.valid_indices.append(i)
        
        print(f"  Found {len(self.valid_indices)}/{len(hf_dataset)} valid examples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def _get_answer(self, answers):
        """
        For eval, deterministically pick the most frequent answer.
        For training, randomly sample for robustness.
        """
        if self.is_eval:
            if isinstance(answers[0], dict) and 'answer' in answers[0]:
                answer_strs = [a['answer'] for a in answers]
            else:
                answer_strs = [str(a) for a in answers]
            counts = Counter(answer_strs)
            return counts.most_common(1)[0][0]
        else:
            if isinstance(answers[0], dict) and 'answer' in answers[0]:
                return random.choice(answers)['answer']
            else:
                return str(random.choice(answers))
    
    def __getitem__(self, idx):
        example = self.dataset[self.valid_indices[idx]]
        
        try:
            image = example['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            question = example['question']
            
            # ============================================
            # [FIX] Response format prompting.
            # LLaVA 1.5 was trained on OK-VQA with this suffix appended.
            # Without it, the model learns that ALL outputs should be
            # 1-2 words, collapsing its generation distribution.
            # With it, brevity is conditional on the instruction,
            # preserving long-form generation ability for other tasks.
            # ============================================
            question = question.strip() + "\nAnswer the question using a single word or phrase."
            
            answer = self._get_answer(example['answers'])
            
            # Build the user turn (question only, no answer) for prefix length calculation
            user_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Full conversation including assistant response
            full_conversation = user_conversation + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            
            # Tokenize user turn with generation prompt to get exact prefix length
            prefix_text = self.processor.apply_chat_template(
                user_conversation, add_generation_prompt=True
            )
            prefix_inputs = self.processor(
                images=image,
                text=prefix_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]
            
            # Tokenize full conversation for training
            full_text = self.processor.apply_chat_template(
                full_conversation, add_generation_prompt=False
            )
            full_inputs = self.processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = full_inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()
            
            # Mask everything up to and including the generation prompt
            labels[:prefix_len] = -100
            
            # Mask image tokens
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            labels[labels == image_token_id] = -100
            
            # Mask pad tokens (but NOT EOS tokens)
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None and pad_token_id != self.processor.tokenizer.eos_token_id:
                labels[labels == pad_token_id] = -100
            
            # ============================================
            # [FIX] Ensure EOS token is present and unmasked.
            # LLaVA's Vicuna tokenizer may not append EOS by default,
            # so the model never learns when to stop generating.
            # This explicitly appends EOS if missing and ensures
            # it is never masked in labels.
            # ============================================
            eos_token_id = self.processor.tokenizer.eos_token_id
            if input_ids[-1] != eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
                labels = torch.cat([labels, torch.tensor([eos_token_id])])
            
            # Safety: ensure any EOS tokens in labels are never masked
            eos_positions = (input_ids == eos_token_id)
            labels[eos_positions] = eos_token_id
            
            return {
                'pixel_values': full_inputs['pixel_values'].squeeze(0),
                'input_ids': input_ids,
                'labels': labels
            }
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(examples):
    """Collate function for DataLoader."""
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


# ============================================
# Model Loading (BF16, no quantization)
# ============================================
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# ============================================
# LoRA Config
# ============================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", 
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("\n" + "="*60)
print("LORA CONFIGURATION")
print("="*60)
print(f"  r (rank):        {lora_config.r}")
print(f"  alpha:           {lora_config.lora_alpha}")
print(f"  alpha/r ratio:   {lora_config.lora_alpha / lora_config.r}")
print(f"  target_modules:  {lora_config.target_modules}")
print(f"  dropout:         {lora_config.lora_dropout}")
print("="*60 + "\n")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

wandb.watch(model, log="all", log_freq=100)


# ============================================
# Dataset Loading
# ============================================
print("Loading OK-VQA datasets...")
print(f"  Train: {TRAIN_DATASET_NAME}")
print(f"  Val: {VAL_DATASET_NAME}")

raw_train_dataset = load_dataset(TRAIN_DATASET_NAME, split="train")
print(f"Loaded {len(raw_train_dataset)} raw training examples")

raw_val_dataset = load_dataset(VAL_DATASET_NAME, split="val2014")
print(f"Loaded {len(raw_val_dataset)} raw val2014 examples")

# ============================================
# Split val2014 into eval (for training) and
# held-out test (for post-training evaluation)
# ============================================
val_indices = list(range(len(raw_val_dataset)))
random.seed(VAL_SPLIT_SEED)
random.shuffle(val_indices)

n_val_eval = int(len(val_indices) * VAL_EVAL_RATIO)
val_eval_indices = val_indices[:n_val_eval]
val_test_indices = val_indices[n_val_eval:]

raw_val_eval = raw_val_dataset.select(val_eval_indices)
raw_val_test = raw_val_dataset.select(val_test_indices)

print(f"  Val2014 split (seed={VAL_SPLIT_SEED}):")
print(f"    Eval during training: {len(raw_val_eval)} ({VAL_EVAL_RATIO:.0%})")
print(f"    Held-out test:        {len(raw_val_test)} ({VAL_TEST_RATIO:.0%})")

print("\nBuilding lazy dataset wrappers...")
train_dataset = VQADataset(raw_train_dataset, processor, MAX_LENGTH, is_eval=False)
val_dataset = VQADataset(raw_val_eval, processor, MAX_LENGTH, is_eval=True)

wandb.log({
    "dataset/total_train_examples": len(raw_train_dataset),
    "dataset/valid_train_examples": len(train_dataset),
    "dataset/train_success_rate": len(train_dataset) / len(raw_train_dataset) * 100,
    "dataset/total_val2014_examples": len(raw_val_dataset),
    "dataset/val_eval_examples": len(raw_val_eval),
    "dataset/val_test_examples": len(raw_val_test),
    "dataset/valid_val_eval_examples": len(val_dataset),
    "dataset/val_eval_success_rate": len(val_dataset) / len(raw_val_eval) * 100,
    "dataset/val_split_seed": VAL_SPLIT_SEED,
})


# ============================================
# Training Configuration
# ============================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    
    # Evaluation settings
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # W&B integration
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"  Learning rate:     {LEARNING_RATE}")
print(f"  Epochs:            {NUM_EPOCHS}")
print(f"  Batch size:        {BATCH_SIZE}")
print(f"  Grad accum steps:  {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch:   {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Warmup ratio:      {WARMUP_RATIO}")
print(f"  Weight decay:      {WEIGHT_DECAY}")
print(f"  Optimizer:         adamw_torch")
print(f"  Quantization:      None (BF16)")
print(f"  LoRA rank:         {lora_config.r}")
print(f"  LoRA alpha:        {lora_config.lora_alpha}")
print(f"  LoRA modules:      {lora_config.target_modules}")
print("="*60 + "\n")

# Create custom callbacks
weight_tracking_callback = WandBWeightTrackingCallback(model, log_freq=50)
gradient_histogram_callback = WandBGradientHistogramCallback(model, log_freq=100)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    callbacks=[weight_tracking_callback, gradient_histogram_callback],
)


# ============================================
# Training
# ============================================
print("\nStarting training...")
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

test_item = raw_val_dataset[0]

try:
    test_image = test_item['image']
    if test_image.mode != 'RGB':
        test_image = test_image.convert('RGB')
    
    test_question = test_item['question']
    
    # Include the same format prompt used during training
    test_question_formatted = test_question.strip() + "\nAnswer the question using a single word or phrase."
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": test_question_formatted}
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
    ground_truth = test_item['answers'][:3]
    
    print(f"\nQuestion: {test_question}")
    print(f"Generated: {generated_text}")
    print(f"Ground truth answers: {ground_truth}")
    
    wandb.log({
        "inference/test_question": test_question,
        "inference/generated_answer": generated_text,
        "inference/ground_truth": str(ground_truth),
        "inference/test_image": wandb.Image(test_image, caption=test_question)
    })

except Exception as e:
    print(f"Inference test failed: {e}")
    import traceback
    traceback.print_exc()

# Finish W&B run
wandb.finish()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model saved to: {OUTPUT_DIR}")
print("="*60)