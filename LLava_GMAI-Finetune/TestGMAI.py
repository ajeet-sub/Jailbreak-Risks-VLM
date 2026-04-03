"""
Evaluate base and fine-tuned LLaVA on GMAI-Reasoning10K held-out test set.

Runs evaluation on both the base LLaVA 1.5-7B and the LoRA fine-tuned
model on the same held-out 5% test set (~500 examples), then prints a
side-by-side comparison with per-modality breakdown.

Usage:
    python eval_gmai.py --model_dir ./llava-gmai-finetuned
    python eval_gmai.py --model_dir ./llava-gmai-finetuned --max_samples 50
    python eval_gmai.py --model_dir ./llava-gmai-finetuned --skip_base
    python eval_gmai.py --model_dir ./llava-gmai-finetuned --eval_split both
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
import argparse
import json
import os
import re
import random
from tqdm import tqdm

# ============================================
# Config
# ============================================
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
HF_DATASET_NAME = "General-Medical-AI/GMAI-Reasoning10K"
HF_DATASET_CONFIG = "reasoning_mcq_rl"
MAX_NEW_TOKENS = 20

# Must match the split parameters from the training script
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05
SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on GMAI-Reasoning10K")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./llava-gmai-finetuned",
        help="Path to the fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL_ID,
        help="Base model ID (default: llava-hf/llava-1.5-7b-hf)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of eval samples (for quick testing)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gmai_eval_results.json",
        help="Path to save detailed results",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "val", "both"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip base model evaluation (only eval fine-tuned)",
    )
    return parser.parse_args()


def normalize_mcq_answer(answer):
    """
    Extract and normalize a multiple-choice letter answer.

    Prioritizes explicit MCQ patterns over grabbing the first
    character, which prevents false matches like "Based" → "B".
    Returns "NO_ANSWER" if no valid letter can be extracted.
    """
    answer = answer.strip()

    # Pattern 1: standalone letter at start (e.g. "A", "B.", "A)")
    # Must NOT be followed by a lowercase letter (rules out "Based", "CT", etc.)
    match = re.match(r"^([A-Da-d])(?:[).\s,:]|$)", answer)
    if match:
        return match.group(1).upper()

    # Pattern 2: letter in parentheses (e.g. "(A)", "(B)")
    match = re.search(r"\(([A-Da-d])\)", answer)
    if match:
        return match.group(1).upper()

    # Pattern 3: "Option X" or "Answer: X" or "Answer is X"
    match = re.search(r"(?:option|answer)[\s:is]*([A-Da-d])\b", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: "correct answer is X" or "correct option is X"
    match = re.search(r"correct\s+(?:answer|option)\s+is\s+([A-Da-d])\b", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 5: letter followed by ) or . mid-string (e.g. "... is C) Neoplastic")
    match = re.search(r"\b([A-Da-d])[).]\s", answer)
    if match:
        return match.group(1).upper()

    # No valid MCQ letter found
    return "NO_ANSWER"


def extract_answer(generated_text):
    """Extract the assistant's response from generated text."""
    if "ASSISTANT:" in generated_text:
        answer = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = generated_text.strip()
    return answer


def recreate_splits(raw_dataset):
    """Recreate the exact same split used during training."""
    indices = list(range(len(raw_dataset)))
    random.seed(SPLIT_SEED)
    random.shuffle(indices)

    n_total = len(indices)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_test = max(1, int(n_total * TEST_RATIO))
    n_train = n_total - n_val - n_test

    train_split = raw_dataset.select(indices[:n_train])
    val_split = raw_dataset.select(indices[n_train : n_train + n_val])
    test_split = raw_dataset.select(indices[n_train + n_val :])

    return train_split, val_split, test_split


def evaluate_model(model, processor, dataset, dataset_dir, model_name, split_name, max_samples=None):
    """
    Run MCQ evaluation on a dataset split with the given model.
    Returns a dict with metrics and per-example results.
    """
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} on {split_name} ({len(dataset)} examples)")
    print(f"{'='*60}")

    model.eval()
    correct = 0
    total = 0
    results = []
    modality_scores = {}

    for i in tqdm(range(len(dataset)), desc=f"Eval [{model_name[:20]}] {split_name}"):
        example = dataset[i]

        try:
            image_path = os.path.join(dataset_dir, example["image"])
            if not os.path.isfile(image_path):
                continue

            image = Image.open(image_path).convert("RGB")

            conversations = example["conversations"]
            question_text = conversations[0]["value"].replace("<image>", "").strip()
            ground_truth = conversations[1]["value"].strip()
            gt_normalized = normalize_mcq_answer(ground_truth)

            # Append format instruction so the base model also
            # knows to respond with just a letter. Without this,
            # the base model produces long-form prose and the
            # letter extractor either fails or grabs noise.
            # The fine-tuned model already learned this format
            # from training, so the instruction is redundant but
            # harmless for it — and essential for a fair comparison.
            if not question_text.endswith("."):
                question_text += "."
            question_text += "\nRespond with only the letter of the correct answer."

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated_text = processor.decode(output[0], skip_special_tokens=True)
            predicted_raw = extract_answer(generated_text)
            predicted_normalized = normalize_mcq_answer(predicted_raw)

            is_correct = predicted_normalized == gt_normalized
            if is_correct:
                correct += 1
            total += 1

            # Track per-modality accuracy
            path_parts = example["image"].split("/")
            modality = path_parts[3] if len(path_parts) >= 4 else "unknown"

            if modality not in modality_scores:
                modality_scores[modality] = {"correct": 0, "total": 0}
            modality_scores[modality]["total"] += 1
            if is_correct:
                modality_scores[modality]["correct"] += 1

            results.append(
                {
                    "index": i,
                    "question": question_text[:150],
                    "predicted_raw": predicted_raw,
                    "predicted": predicted_normalized,
                    "ground_truth": gt_normalized,
                    "correct": is_correct,
                    "modality": modality,
                }
            )

            if (i + 1) % 100 == 0:
                running_acc = correct / total * 100
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {running_acc:.2f}%")

        except Exception as e:
            print(f"  Error on example {i}: {e}")
            total += 1
            results.append(
                {
                    "index": i,
                    "question": "",
                    "predicted_raw": "ERROR",
                    "predicted": "ERROR",
                    "ground_truth": "",
                    "correct": False,
                    "modality": "unknown",
                }
            )

    accuracy = correct / total * 100 if total > 0 else 0
    format_failures = sum(1 for r in results if r["predicted"] == "NO_ANSWER")
    format_failure_pct = format_failures / total * 100 if total > 0 else 0

    modality_results = {}
    for mod, scores in sorted(modality_scores.items()):
        mod_acc = scores["correct"] / scores["total"] * 100 if scores["total"] > 0 else 0
        modality_results[mod] = {
            "accuracy": mod_acc,
            "correct": scores["correct"],
            "total": scores["total"],
        }

    print(f"\n  {model_name} — Accuracy: {accuracy:.2f}% (format failures: {format_failures}/{total})")

    return {
        "model_name": model_name,
        "split": split_name,
        "num_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_failures": format_failures,
        "format_failure_pct": format_failure_pct,
        "per_modality": modality_results,
        "per_example": results,
    }


def main():
    args = parse_args()

    # ---- Download dataset ----
    print(f"Downloading dataset: {HF_DATASET_NAME}...")
    dataset_dir = snapshot_download(
        repo_id=HF_DATASET_NAME,
        repo_type="dataset",
    )
    print(f"Dataset cached at: {dataset_dir}")

    # ---- Load and split ----
    raw_dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split="train")
    print(f"Loaded {len(raw_dataset)} total examples")

    train_split, val_split, test_split = recreate_splits(raw_dataset)
    print(f"Splits: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test")

    # Determine which splits to evaluate
    eval_splits = {}
    if args.eval_split in ("test", "both"):
        eval_splits["test"] = test_split
    if args.eval_split in ("val", "both"):
        eval_splits["val"] = val_split

    # ---- Load base model ----
    print(f"\nLoading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "right"

    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    all_results = {}

    # ---- Evaluate base model on each split ----
    if not args.skip_base:
        for split_name, split_data in eval_splits.items():
            key = f"base_{split_name}"
            all_results[key] = evaluate_model(
                base_model, processor, split_data, dataset_dir,
                model_name="LLaVA-1.5-7B (base)",
                split_name=split_name,
                max_samples=args.max_samples,
            )

    # ---- Load LoRA adapter and evaluate fine-tuned model ----
    print(f"\nLoading LoRA adapter from: {args.model_dir}")
    finetuned_model = PeftModel.from_pretrained(base_model, args.model_dir)

    for split_name, split_data in eval_splits.items():
        key = f"finetuned_{split_name}"
        all_results[key] = evaluate_model(
            finetuned_model, processor, split_data, dataset_dir,
            model_name="LLaVA-1.5-7B + LoRA (fine-tuned)",
            split_name=split_name,
            max_samples=args.max_samples,
        )

    # ---- Print comparison ----
    print("\n" + "=" * 60)
    print("GMAI-REASONING10K EVALUATION COMPARISON")
    print("=" * 60)
    print(f"  Dataset:  {HF_DATASET_NAME} ({HF_DATASET_CONFIG})")
    print(f"  Adapter:  {args.model_dir}")

    for split_name in eval_splits:
        base_key = f"base_{split_name}"
        ft_key = f"finetuned_{split_name}"

        print(f"\n  --- {split_name.upper()} SPLIT ---")
        print(f"  {'Metric':<25s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")

        ft_acc = all_results[ft_key]["accuracy"]
        ft_n = all_results[ft_key]["num_examples"]

        if base_key in all_results:
            base_acc = all_results[base_key]["accuracy"]
            delta = ft_acc - base_acc
            base_ff = all_results[base_key]["format_failure_pct"]
            ft_ff = all_results[ft_key]["format_failure_pct"]
            delta_ff = ft_ff - base_ff
            print(f"  {'Accuracy':<25s} {base_acc:>9.2f}% {ft_acc:>11.2f}% {delta:>+9.2f}%")
            print(f"  {'Format failures':<25s} {base_ff:>9.2f}% {ft_ff:>11.2f}% {delta_ff:>+9.2f}%")
            print(f"  {'Num examples':<25s} {all_results[base_key]['num_examples']:>10d} {ft_n:>12d}")

            # Per-modality comparison
            base_mods = all_results[base_key]["per_modality"]
            ft_mods = all_results[ft_key]["per_modality"]
            all_mods = sorted(
                set(list(base_mods.keys()) + list(ft_mods.keys())),
                key=lambda m: ft_mods.get(m, {}).get("total", 0),
                reverse=True,
            )

            if all_mods:
                print(f"\n  Per-modality breakdown ({split_name}):")
                print(f"  {'Modality':<30s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s} {'N':>5s}")
                print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10} {'-'*5}")

                for mod in all_mods:
                    b_acc = base_mods.get(mod, {}).get("accuracy", 0)
                    f_acc = ft_mods.get(mod, {}).get("accuracy", 0)
                    d_acc = f_acc - b_acc
                    n = ft_mods.get(mod, {}).get("total", 0)
                    print(f"  {mod:<30s} {b_acc:>9.2f}% {f_acc:>11.2f}% {d_acc:>+9.2f}% {n:>5d}")
        else:
            print(f"  {'Accuracy':<25s} {'(skipped)':>10s} {ft_acc:>11.2f}%")
            print(f"  {'Num examples':<25s} {'':>10s} {ft_n:>12d}")

    print("\n" + "=" * 60)

    # ---- Save results ----
    output = {
        "dataset": HF_DATASET_NAME,
        "config": HF_DATASET_CONFIG,
        "model_dir": args.model_dir,
        "split_seed": SPLIT_SEED,
        "results": {
            k: {key: val for key, val in v.items() if key != "per_example"}
            for k, v in all_results.items()
        },
        "detailed_results": {k: v["per_example"] for k, v in all_results.items()},
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {args.output_file}")

    # ---- Print sample head-to-head comparisons ----
    for split_name in eval_splits:
        base_key = f"base_{split_name}"
        ft_key = f"finetuned_{split_name}"

        if base_key in all_results:
            print(f"\nSample head-to-head comparisons ({split_name}):")
            base_examples = all_results[base_key]["per_example"]
            ft_examples = all_results[ft_key]["per_example"]

            for j in range(min(10, len(base_examples))):
                b = base_examples[j]
                f_ = ft_examples[j]
                b_status = "✓" if b["correct"] else "✗"
                f_status = "✓" if f_["correct"] else "✗"
                print(f"  [{b['modality']}] Q: {b['question'][:60]}...")
                print(f"    Base:       {b_status} {b['predicted']:<5s}  GT: {b['ground_truth']}")
                print(f"    Fine-tuned: {f_status} {f_['predicted']:<5s}")
                print()


if __name__ == "__main__":
    main()