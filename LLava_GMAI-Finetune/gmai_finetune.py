import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
import random
import os
import numpy as np
import wandb
from torch.utils.data import Dataset

# ============================================
# Config
# ============================================
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Dataset: GMAI-Reasoning10K from HuggingFace
# Uses the "reasoning_mcq_rl" config (10K multiple-choice
# medical VQA with short answers like "A", "B", etc.)
# Same human/gpt conversation format as GMAI-VL.
HF_DATASET_NAME = "General-Medical-AI/GMAI-Reasoning10K"
HF_DATASET_CONFIG = "reasoning_mcq_rl"

OUTPUT_DIR = "./llava-gmai-finetuned"

# Training hyperparameters — identical to OK-VQA script
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 32  # effective batch size = 4 * 32 = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_LENGTH = 2048

# Split ratios (dataset has no predefined splits)
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05
SPLIT_SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================
# W&B init
# ============================================
WANDB_PROJECT = "llava-gmai-lora"
WANDB_RUN_NAME = "llava-1.5-7b-lora-gmai-reasoning10k"

wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model_id": MODEL_ID,
        "dataset": HF_DATASET_NAME,
        "dataset_config": HF_DATASET_CONFIG,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": "cosine",
        "max_length": MAX_LENGTH,
        "quantization": "none",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    },
)


# ============================================
# Model Loading (BF16, no quantization)
# ============================================
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ============================================
# LoRA Config — identical to OK-VQA script
# ============================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("\n" + "=" * 60)
print("LORA CONFIGURATION")
print("=" * 60)
print(f"  r (rank):        {lora_config.r}")
print(f"  alpha:           {lora_config.lora_alpha}")
print(f"  alpha/r ratio:   {lora_config.lora_alpha / lora_config.r}")
print(f"  target_modules:  {lora_config.target_modules}")
print(f"  dropout:         {lora_config.lora_dropout}")
print("=" * 60 + "\n")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

wandb.watch(model, log="all", log_freq=100)


# ============================================
# W&B Callbacks (identical to OK-VQA script)
# ============================================
class WandBWeightTrackingCallback(TrainerCallback):
    """Track LoRA weight statistics and movement during training."""

    def __init__(self, model, log_freq=50):
        self.model = model
        self.log_freq = log_freq
        self.initial_weights = {}
        self.previous_weights = {}
        self._store_initial_weights()

    def _get_lora_params(self):
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_params[name] = param
        return lora_params

    def _get_module_type(self, name):
        for module in ["q_proj", "v_proj"]:
            if module in name:
                return module
        return "other"

    def _get_layer_idx(self, name):
        import re
        match = re.search(r"layers\.(\d+)\.", name)
        return int(match.group(1)) if match else -1

    def _store_initial_weights(self):
        for name, param in self._get_lora_params().items():
            self.initial_weights[name] = param.detach().clone().cpu()
            self.previous_weights[name] = param.detach().clone().cpu()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq != 0:
            return

        lora_params = self._get_lora_params()

        weight_stats = {
            "lora_A": {"norms": [], "grad_norms": [], "changes": [], "total_changes": []},
            "lora_B": {"norms": [], "grad_norms": [], "changes": [], "total_changes": []},
        }

        module_stats = {
            mod: {"norms": [], "grad_norms": [], "total_changes": []}
            for mod in ["q_proj", "v_proj"]
        }

        all_weight_norms = []
        all_grad_norms = []
        all_step_changes = []
        all_total_changes = []

        for name, param in lora_params.items():
            lora_type = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else None
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
                weight_stats[lora_type]["norms"].append(weight_norm)
                weight_stats[lora_type]["grad_norms"].append(grad_norm)
                weight_stats[lora_type]["changes"].append(step_change)
                weight_stats[lora_type]["total_changes"].append(total_change)

            if module_type in module_stats:
                module_stats[module_type]["norms"].append(weight_norm)
                module_stats[module_type]["grad_norms"].append(grad_norm)
                module_stats[module_type]["total_changes"].append(total_change)

        log_dict = {
            "weights/mean_weight_norm": np.mean(all_weight_norms) if all_weight_norms else 0,
            "weights/max_weight_norm": np.max(all_weight_norms) if all_weight_norms else 0,
            "gradients/mean_grad_norm": np.mean(all_grad_norms) if all_grad_norms else 0,
            "weight_delta/mean_step_change": np.mean(all_step_changes) if all_step_changes else 0,
            "weight_movement/mean_total_change": np.mean(all_total_changes) if all_total_changes else 0,
            "weight_movement/sum_total_change": np.sum(all_total_changes) if all_total_changes else 0,
        }

        for lora_type in ["lora_A", "lora_B"]:
            stats = weight_stats[lora_type]
            if stats["norms"]:
                log_dict[f"lora_weights/{lora_type}_mean_norm"] = np.mean(stats["norms"])
                log_dict[f"lora_gradients/{lora_type}_mean_grad_norm"] = np.mean(stats["grad_norms"])
            if stats["total_changes"]:
                log_dict[f"lora_movement/{lora_type}_mean_total_change"] = np.mean(stats["total_changes"])

        for module_type, stats in module_stats.items():
            if stats["norms"]:
                log_dict[f"module_weights/{module_type}_mean_norm"] = np.mean(stats["norms"])
            if stats["total_changes"]:
                log_dict[f"module_movement/{module_type}_total_change"] = np.mean(stats["total_changes"])

        wandb.log(log_dict, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        lora_params = self._get_lora_params()

        layer_movements = {}
        module_movements = {mod: [] for mod in ["q_proj", "v_proj"]}

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

        wandb.log(
            {
                f"epoch_{int(state.epoch)}/layer_weight_movements": wandb.plot.bar(
                    layer_table, "Layer", "Weight Movement",
                    title=f"Weight Movement by Layer (Epoch {int(state.epoch)})",
                ),
                f"epoch_{int(state.epoch)}/module_weight_movements": wandb.plot.bar(
                    module_table, "Module", "Weight Movement",
                    title=f"Weight Movement by Module Type (Epoch {int(state.epoch)})",
                ),
                f"epoch_summary/epoch_{int(state.epoch)}_total_movement": sum(
                    np.mean(v) for v in layer_movements.values()
                ),
            },
            step=state.global_step,
        )


class WandBGradientHistogramCallback(TrainerCallback):
    """Log gradient histograms periodically."""

    def __init__(self, model, log_freq=100):
        self.model = model
        self.log_freq = log_freq

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq != 0:
            return

        all_grads = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad and param.grad is not None:
                all_grads.extend(param.grad.detach().cpu().flatten().numpy())

        if all_grads:
            wandb.log(
                {"gradients/histogram": wandb.Histogram(all_grads)},
                step=state.global_step,
            )


# ============================================
# Dataset Loading from HuggingFace
# ============================================
# GMAI-Reasoning10K stores images as file paths
# within the dataset repo. We use snapshot_download
# to pull the full repo (including images), then
# resolve paths locally.
# ============================================
print(f"Downloading dataset: {HF_DATASET_NAME} ({HF_DATASET_CONFIG})...")
print("This will download ~4GB of images on first run (cached after).\n")

# Download the full dataset repo (images + JSONL)
dataset_dir = snapshot_download(
    repo_id=HF_DATASET_NAME,
    repo_type="dataset",
)
print(f"Dataset cached at: {dataset_dir}")

# Load the JSONL metadata via HuggingFace datasets
raw_dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split="train")
print(f"Loaded {len(raw_dataset)} examples from {HF_DATASET_CONFIG}")

# Debug first entry
print(f"\nFirst example:")
print(f"  Image path: {raw_dataset[0]['image']}")
print(f"  Num turns:  {len(raw_dataset[0]['conversations'])}")
print(f"  Human turn: {raw_dataset[0]['conversations'][0]['value'][:100]}...")
print(f"  GPT turn:   {raw_dataset[0]['conversations'][1]['value']}")
print()


# ============================================
# Manual Train / Val / Test Split
# ============================================
# Dataset only has a single "train" split, so
# we create our own 85/10/5 split with a fixed
# seed for reproducibility.
# ============================================
print("Creating train/val/test splits...")
indices = list(range(len(raw_dataset)))
random.seed(SPLIT_SEED)
random.shuffle(indices)

n_total = len(indices)
n_val = max(1, int(n_total * VAL_RATIO))
n_test = max(1, int(n_total * TEST_RATIO))
n_train = n_total - n_val - n_test

train_indices = indices[:n_train]
val_indices = indices[n_train : n_train + n_val]
test_indices = indices[n_train + n_val :]

train_split = raw_dataset.select(train_indices)
val_split = raw_dataset.select(val_indices)
test_split = raw_dataset.select(test_indices)

print(f"  Train: {len(train_split)}")
print(f"  Val:   {len(val_split)}")
print(f"  Test:  {len(test_split)} (held out for post-training eval)")
print()


# ============================================
# Conversation format conversion
# ============================================
def conversations_to_chat_format(conversations):
    """
    Convert GMAI-Reasoning10K conversation list into the
    HuggingFace chat message format for apply_chat_template.

    The RL config uses:
        {"from": "human", "value": "<image>\n...question..."}
        {"from": "gpt",   "value": "A"}

    Returns (user_messages, full_messages, assistant_text).
    """
    user_messages = []
    full_messages = []
    assistant_text = ""

    for i, turn in enumerate(conversations):
        role = turn["from"]
        text = turn["value"]

        # Strip <image> placeholder — processor handles image injection
        text = text.replace("<image>", "").strip()

        if role == "human":
            # [FIX] Append format prompt so brevity is conditional on the
            # instruction, not a globally learned behavior. Without this,
            # the model collapses all outputs to 1-2 tokens since GMAI
            # answers are single letters (A/B/C/D).
            text = text + "\nAnswer the question using a single word or phrase."
            msg = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
            user_messages.append(msg)
            full_messages.append(msg)

        elif role == "gpt":
            msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
            full_messages.append(msg)
            if i == len(conversations) - 1:
                assistant_text = text

    return user_messages, full_messages, assistant_text


# ============================================
# Lazy Dataset
# ============================================
class GMAIReasoningDataset(Dataset):
    """
    Lazy PyTorch Dataset for GMAI-Reasoning10K fine-tuning.

    Wraps a HuggingFace dataset split. Images are loaded from
    the snapshot_download cache directory at __getitem__ time.
    """

    def __init__(self, hf_dataset, dataset_dir, processor, max_length=2048, is_eval=False):
        self.dataset = hf_dataset
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.max_length = max_length
        self.is_eval = is_eval

        # Validate which examples have loadable images
        self.valid_indices = []
        for i in range(len(hf_dataset)):
            image_path = os.path.join(dataset_dir, hf_dataset[i]["image"])
            convs = hf_dataset[i]["conversations"]
            has_human = any(t["from"] == "human" for t in convs)
            has_gpt = any(t["from"] == "gpt" for t in convs)

            if has_human and has_gpt and os.path.isfile(image_path):
                self.valid_indices.append(i)

        print(f"  Found {len(self.valid_indices)}/{len(hf_dataset)} valid examples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        example = self.dataset[self.valid_indices[idx]]

        try:
            # ---- Load image ----
            image_path = os.path.join(self.dataset_dir, example["image"])
            image = Image.open(image_path).convert("RGB")

            # ---- Convert conversations to chat format ----
            conversations = example["conversations"]
            user_messages, full_messages, _ = conversations_to_chat_format(conversations)

            # ---- Tokenize user turn with generation prompt for prefix length ----
            prefix_text = self.processor.apply_chat_template(
                user_messages, add_generation_prompt=True
            )
            prefix_inputs = self.processor(
                images=image,
                text=prefix_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]

            # ---- Tokenize full conversation for training ----
            full_text = self.processor.apply_chat_template(
                full_messages, add_generation_prompt=False
            )
            full_inputs = self.processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = full_inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            # ---- Mask everything up to and including the generation prompt ----
            labels[:prefix_len] = -100

            # ---- Mask image tokens ----
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            labels[labels == image_token_id] = -100

            # ---- Mask pad tokens (but NOT EOS tokens) ----
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None and pad_token_id != self.processor.tokenizer.eos_token_id:
                labels[labels == pad_token_id] = -100

            # ---- EOS token enforcement ----
            eos_token_id = self.processor.tokenizer.eos_token_id
            if input_ids[-1] != eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
                labels = torch.cat([labels, torch.tensor([eos_token_id])])

            # Ensure any EOS tokens in labels are never masked
            eos_positions = input_ids == eos_token_id
            labels[eos_positions] = eos_token_id

            return {
                "pixel_values": full_inputs["pixel_values"].squeeze(0),
                "input_ids": input_ids,
                "labels": labels,
            }

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# ============================================
# Collate
# ============================================
def collate_fn(examples):
    examples = [ex for ex in examples if ex is not None]
    if len(examples) == 0:
        return None

    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])

    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]
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
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100)]))
        attention_masks.append(torch.cat([torch.ones(seq_len), torch.zeros(pad_len)]))

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_labels),
    }


# ============================================
# Build dataset wrappers
# ============================================
print("Building lazy dataset wrappers...")
train_dataset = GMAIReasoningDataset(
    hf_dataset=train_split,
    dataset_dir=dataset_dir,
    processor=processor,
    max_length=MAX_LENGTH,
    is_eval=False,
)
val_dataset = GMAIReasoningDataset(
    hf_dataset=val_split,
    dataset_dir=dataset_dir,
    processor=processor,
    max_length=MAX_LENGTH,
    is_eval=True,
)

wandb.log({
    "dataset/total_examples": len(raw_dataset),
    "dataset/train_examples": len(train_split),
    "dataset/val_examples": len(val_split),
    "dataset/test_examples": len(test_split),
    "dataset/valid_train_examples": len(train_dataset),
    "dataset/valid_val_examples": len(val_dataset),
    "dataset/train_success_rate": len(train_dataset) / max(1, len(train_split)) * 100,
    "dataset/val_success_rate": len(val_dataset) / max(1, len(val_split)) * 100,
})


# ============================================
# Training Configuration — identical to OK-VQA
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

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)
print(f"  Learning rate:     {LEARNING_RATE}")
print(f"  Epochs:            {NUM_EPOCHS}")
print(f"  Batch size:        {BATCH_SIZE}")
print(f"  Grad accum steps:  {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch:   {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Warmup ratio:      {WARMUP_RATIO}")
print(f"  Weight decay:      {WEIGHT_DECAY}")
print(f"  Optimizer:         adamw_torch")
print(f"  LR scheduler:      cosine")
print(f"  Quantization:      None (BF16)")
print(f"  Max grad norm:     1.0")
print(f"  LoRA rank:         {lora_config.r}")
print(f"  LoRA alpha:        {lora_config.lora_alpha}")
print(f"  LoRA modules:      {lora_config.target_modules}")
print(f"  Train examples:    {len(train_dataset)}")
print(f"  Val examples:      {len(val_dataset)}")
print("=" * 60 + "\n")

# Create callbacks
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
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Log model artifact to W&B
artifact = wandb.Artifact(
    name="llava-gmai-reasoning-lora",
    type="model",
    description="LoRA fine-tuned LLaVA on GMAI-Reasoning10K (medical MCQ VQA)",
)
artifact.add_dir(OUTPUT_DIR)
wandb.log_artifact(artifact)


# ============================================
# Inference Test
# ============================================
print("\nTesting inference...")
model.eval()

test_example = val_split[0]
test_image_path = os.path.join(dataset_dir, test_example["image"])

try:
    test_image = Image.open(test_image_path).convert("RGB")

    # Use the human turn as the test prompt
    first_human = test_example["conversations"][0]["value"].replace("<image>", "").strip()
    ground_truth = test_example["conversations"][1]["value"]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": first_human},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_text = processor.decode(output[0], skip_special_tokens=True)
    print(f"\nQuestion:     {first_human[:120]}...")
    print(f"Generated:    {generated_text}")
    print(f"Ground truth: {ground_truth}")

    wandb.log(
        {
            "inference/test_question": first_human,
            "inference/generated_answer": generated_text,
            "inference/ground_truth": ground_truth,
            "inference/test_image": wandb.Image(test_image, caption=first_human[:100]),
        }
    )

except Exception as e:
    print(f"Inference test failed: {e}")
    import traceback
    traceback.print_exc()

# Finish W&B run
wandb.finish()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved to: {OUTPUT_DIR}")
print(f"Test set ({len(test_split)} examples) held out for post-training evaluation.")
print("=" * 60)