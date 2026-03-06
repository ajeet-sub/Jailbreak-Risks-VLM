import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import json
import random
import os
import wandb
from torch.utils.data import Dataset

# -----------------------
# Config
# -----------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Path to your local GMAI-VL data.
# The dataset uses the standard LLaVA JSONL format:
#   {"id": "...", "image": "relative/path/to/image.jpg", "conversations": [
#       {"from": "human", "value": "<image>\nDescribe this image."},
#       {"from": "gpt",   "value": "This is an X-ray showing..."}
#   ]}
# Download the dataset from: https://github.com/uni-medical/GMAI-VL
# (dataset release is forthcoming — check the repo for updates)
ANNOTATIONS_FILE = "./data/gmai_vl/annotations.jsonl"  # path to your JSONL file
IMAGE_DIR = "./data/gmai_vl/images"                     # root dir for images

OUTPUT_DIR = "./llava-gmai-finetuned"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 2048
MAX_SAMPLES = 10000  # set to None to use all samples

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------
# W&B init
# -----------------------
wandb.init(
    project="llava-gmai-lora",
    name="llava_gmai_vl_run",
)

# -----------------------
# Model + Processor
# -----------------------
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------
# Save initial trainable (LoRA) weights for movement tracking
# -----------------------
initial_trainable_params = {
    name: p.detach().clone().cpu()
    for name, p in model.named_parameters()
    if p.requires_grad
}

# -----------------------
# W&B callback to track weight movement
# -----------------------
class WeightMovementCallback(TrainerCallback):
    def __init__(self, initial_params, log_every=10):
        self.initial_params = initial_params
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return

        model = kwargs["model"]
        total_delta_sq = 0.0
        total_norm_sq = 0.0
        count = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            current = p.detach().float().cpu().view(-1)
            initial = self.initial_params[name].float().view(-1)
            delta = current - initial
            total_delta_sq += torch.sum(delta * delta).item()
            total_norm_sq += torch.sum(current * current).item()
            count += 1

        if count == 0:
            return

        wandb.log(
            {
                "weight_delta_l2": total_delta_sq ** 0.5,
                "weight_norm_l2": total_norm_sq ** 0.5,
            },
            step=state.global_step,
        )

# -----------------------
# Dataset
# -----------------------

def load_annotations(annotations_file, max_samples=None):
    """
    Load GMAI-VL JSONL annotations.

    Each line is a JSON object with the LLaVA conversation format:
        {
            "id": "sample_001",
            "image": "relative/path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat abnormality is visible?"},
                {"from": "gpt",   "value": "There is a nodule in the right lung..."}
            ]
        }

    For multi-turn conversations the turns simply alternate human/gpt.
    """
    samples = []
    with open(annotations_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def conversations_to_llava_prompt(conversations):
    """
    Convert GMAI-VL style conversations list into a single LLaVA prompt string
    and extract the final assistant response for label masking.

    Returns (prompt_text, assistant_response) where prompt_text includes
    everything up to and including 'ASSISTANT:' and assistant_response is
    the target text.
    """
    # Build full text from conversation turns
    # LLaVA 1.5 chat template: USER: ... ASSISTANT: ...
    prompt_parts = []
    assistant_response = ""

    turns = list(conversations)  # copy so we can pop
    for i, turn in enumerate(turns):
        role = turn["from"]           # "human" or "gpt"
        text = turn["value"]

        # Strip the <image> placeholder — processor handles image injection
        text = text.replace("<image>", "").strip()

        if role == "human":
            prompt_parts.append(f"USER: {text}")
        elif role == "gpt":
            if i == len(turns) - 1:
                # Last gpt turn = target response
                assistant_response = text
                prompt_parts.append(f"ASSISTANT: {text}")
            else:
                prompt_parts.append(f"ASSISTANT: {text}")

    full_text = "\n".join(prompt_parts)
    return full_text, assistant_response


class GMAIDataset(Dataset):
    """
    PyTorch Dataset for GMAI-VL fine-tuning.

    Loads images from local disk and tokenises conversations
    using the LLaVA processor.
    """

    def __init__(self, annotations, image_dir, processor, max_length=2048):
        self.annotations = annotations
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # Pre-process all samples and cache valid ones
        print("Pre-processing dataset...")
        self.processed = []
        for i, ann in enumerate(annotations):
            result = self._process(ann)
            if result is not None:
                self.processed.append(result)
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(annotations)}, "
                      f"valid: {len(self.processed)}")
        print(f"Total valid examples: {len(self.processed)}")

    def _process(self, ann):
        try:
            # ---- Load image ----
            image_rel = ann.get("image", "")
            image_path = os.path.join(self.image_dir, image_rel)
            image = Image.open(image_path).convert("RGB")

            # ---- Build text from conversations ----
            conversations = ann.get("conversations", [])
            if len(conversations) < 2:
                return None  # need at least one human + one gpt turn

            full_text, _ = conversations_to_llava_prompt(conversations)

            # ---- Tokenise ----
            inputs = self.processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            # ---- Build labels: mask everything before ASSISTANT: ----
            labels = inputs["input_ids"].clone()

            # Mask padding
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            # Mask image tokens
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            labels[labels == image_token_id] = -100

            # Mask the prompt (everything up to and including "ASSISTANT:")
            assistant_start = full_text.find("ASSISTANT:")
            if assistant_start != -1:
                prefix_text = full_text[:assistant_start + len("ASSISTANT:")]
                prefix_tokens = self.processor.tokenizer(
                    prefix_text, add_special_tokens=False
                )["input_ids"]
                labels[0, :len(prefix_tokens)] = -100

            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "input_ids": inputs["input_ids"].squeeze(0),
                "labels": labels.squeeze(0),
            }

        except Exception as e:
            return None  # skip broken examples silently

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]


# -----------------------
# Collate
# -----------------------
def collate_fn(examples):
    examples = [ex for ex in examples if ex is not None]
    if not examples:
        return None

    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])

    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]
    max_len = max(ids.shape[0] for ids in input_ids)

    padded_input_ids, padded_labels, attention_masks = [], [], []

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


# -----------------------
# Load data
# -----------------------
print(f"Loading annotations from: {ANNOTATIONS_FILE}")
annotations = load_annotations(ANNOTATIONS_FILE, max_samples=MAX_SAMPLES)
print(f"Loaded {len(annotations)} annotation entries.")

# Debug first entry
if annotations:
    first = annotations[0]
    print(f"\nFirst example keys: {list(first.keys())}")
    print(f"Image path: {first.get('image', 'N/A')}")
    print(f"Num conversation turns: {len(first.get('conversations', []))}")
    print(f"First turn: {first['conversations'][0] if first.get('conversations') else 'N/A'}")
    print()

train_dataset = GMAIDataset(
    annotations=annotations,
    image_dir=IMAGE_DIR,
    processor=processor,
    max_length=MAX_LENGTH,
)

# -----------------------
# Training args
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=50,
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
    report_to=["wandb"],
    run_name="llava_gmai_vl_run",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    callbacks=[WeightMovementCallback(initial_trainable_params, log_every=10)],
)

print("Starting training...")
trainer.train()

# -----------------------
# Save
# -----------------------
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# -----------------------
# Quick inference test
# -----------------------
print("\nTesting inference...")
model.eval()

if annotations:
    test_ann = annotations[0]
    test_image_path = os.path.join(IMAGE_DIR, test_ann.get("image", ""))
    try:
        test_image = Image.open(test_image_path).convert("RGB")

        # Use the first human turn as the test prompt
        first_human = next(
            (t["value"].replace("<image>", "").strip()
             for t in test_ann["conversations"] if t["from"] == "human"),
            "Describe this medical image."
        )

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": first_human},
            ],
        }]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_text = processor.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt : {first_human}")
        print(f"Generated: {generated_text}")

    except Exception as e:
        print(f"Inference test failed: {e}")

print("\nTraining complete!")
wandb.finish()