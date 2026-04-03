"""
Evaluate base and fine-tuned LLaVA on OK-VQA held-out test set.

Runs evaluation on both the base LLaVA 1.5-7B and the LoRA fine-tuned
model on the same held-out 40% of val2014, then prints a side-by-side
comparison.

Usage:
    python eval_okvqa.py --model_dir ./llava-okvqa-finetuned
    python eval_okvqa.py --model_dir ./llava-okvqa-finetuned --max_samples 50
    python eval_okvqa.py --model_dir ./llava-okvqa-finetuned --skip_base
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
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
VAL_DATASET_NAME = "lmms-lab/OK-VQA"
VAL_SPLIT = "val2014"
MAX_NEW_TOKENS = 50

# Must match the split parameters from the training script
VAL_EVAL_RATIO = 0.6
VAL_SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on OK-VQA")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./llava-okvqa-finetuned",
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
        default="okvqa_eval_results.json",
        help="Path to save detailed results",
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip base model evaluation (only eval fine-tuned)",
    )
    return parser.parse_args()


def normalize_answer(answer):
    """Normalize an answer string for comparison."""
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\b(a|an|the)\b", "", answer)
    answer = " ".join(answer.split())
    return answer


def vqa_soft_accuracy(predicted, ground_truth_answers):
    """Official VQA soft accuracy: min(matching_annotators / 3, 1.0)."""
    pred_normalized = normalize_answer(predicted)

    if isinstance(ground_truth_answers[0], dict) and "answer" in ground_truth_answers[0]:
        gt_answers = [a["answer"] for a in ground_truth_answers]
    else:
        gt_answers = [str(a) for a in ground_truth_answers]

    gt_normalized = [normalize_answer(a) for a in gt_answers]
    match_count = sum(1 for gt in gt_normalized if gt == pred_normalized)
    return min(match_count / 3.0, 1.0)


def extract_answer(generated_text):
    """Extract the assistant's response from generated text."""
    if "ASSISTANT:" in generated_text:
        answer = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = generated_text.strip()
    return answer


def evaluate_model(model, processor, dataset, model_name):
    """
    Run VQA evaluation on a dataset with the given model.
    Returns a dict with metrics and per-example results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model.eval()
    scores = []
    results = []

    for i in tqdm(range(len(dataset)), desc=f"Eval [{model_name}]"):
        example = dataset[i]

        try:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            question = example["question"].strip()
            question_formatted = question + "\nAnswer the question using a single word or phrase."

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_formatted},
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
            predicted_answer = extract_answer(generated_text)

            score = vqa_soft_accuracy(predicted_answer, example["answers"])
            scores.append(score)

            results.append(
                {
                    "question_id": example.get("question_id", i),
                    "question": question,
                    "predicted": predicted_answer,
                    "ground_truth": [
                        a["answer"] if isinstance(a, dict) else str(a)
                        for a in example["answers"][:5]
                    ],
                    "score": score,
                }
            )

            if (i + 1) % 100 == 0:
                running_acc = sum(scores) / len(scores) * 100
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {running_acc:.2f}%")

        except Exception as e:
            print(f"  Error on example {i}: {e}")
            scores.append(0.0)
            results.append(
                {
                    "question_id": example.get("question_id", i),
                    "question": example.get("question", ""),
                    "predicted": "ERROR",
                    "ground_truth": [],
                    "score": 0.0,
                }
            )

    overall_accuracy = sum(scores) / len(scores) * 100
    exact_match = sum(1 for s in scores if s >= 1.0) / len(scores) * 100
    partial_match = sum(1 for s in scores if 0 < s < 1.0) / len(scores) * 100
    no_match = sum(1 for s in scores if s == 0.0) / len(scores) * 100

    print(f"\n  {model_name} — VQA Accuracy: {overall_accuracy:.2f}%")

    return {
        "model_name": model_name,
        "num_examples": len(scores),
        "vqa_accuracy": overall_accuracy,
        "exact_match_pct": exact_match,
        "partial_match_pct": partial_match,
        "no_match_pct": no_match,
        "per_example": results,
    }


def main():
    args = parse_args()

    # ---- Load dataset and recreate split ----
    print(f"Loading {VAL_DATASET_NAME} ({VAL_SPLIT})...")
    raw_val_dataset = load_dataset(VAL_DATASET_NAME, split=VAL_SPLIT)
    print(f"Loaded {len(raw_val_dataset)} total val2014 examples")

    val_indices = list(range(len(raw_val_dataset)))
    random.seed(VAL_SPLIT_SEED)
    random.shuffle(val_indices)

    n_val_eval = int(len(val_indices) * VAL_EVAL_RATIO)
    val_test_indices = val_indices[n_val_eval:]

    dataset = raw_val_dataset.select(val_test_indices)
    print(f"Held-out test portion: {len(dataset)} examples (seed={VAL_SPLIT_SEED})")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} examples")

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

    # ---- Evaluate base model ----
    if not args.skip_base:
        base_results = evaluate_model(
            base_model, processor, dataset,
            model_name="LLaVA-1.5-7B (base)"
        )
        all_results["base"] = base_results

    # ---- Load LoRA adapter and evaluate fine-tuned model ----
    print(f"\nLoading LoRA adapter from: {args.model_dir}")
    finetuned_model = PeftModel.from_pretrained(base_model, args.model_dir)

    finetuned_results = evaluate_model(
        finetuned_model, processor, dataset,
        model_name="LLaVA-1.5-7B + LoRA (fine-tuned)"
    )
    all_results["finetuned"] = finetuned_results

    # ---- Print comparison ----
    print("\n" + "=" * 60)
    print("OK-VQA EVALUATION COMPARISON")
    print("=" * 60)
    print(f"  Dataset:       {VAL_DATASET_NAME} (held-out {1-VAL_EVAL_RATIO:.0%} of {VAL_SPLIT})")
    print(f"  Num examples:  {len(dataset)}")
    print(f"  Adapter:       {args.model_dir}")
    print()
    print(f"  {'Metric':<25s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")

    if "base" in all_results:
        base_acc = all_results["base"]["vqa_accuracy"]
        ft_acc = all_results["finetuned"]["vqa_accuracy"]
        delta_acc = ft_acc - base_acc

        base_em = all_results["base"]["exact_match_pct"]
        ft_em = all_results["finetuned"]["exact_match_pct"]
        delta_em = ft_em - base_em

        base_nm = all_results["base"]["no_match_pct"]
        ft_nm = all_results["finetuned"]["no_match_pct"]
        delta_nm = ft_nm - base_nm

        print(f"  {'VQA Accuracy':<25s} {base_acc:>9.2f}% {ft_acc:>11.2f}% {delta_acc:>+9.2f}%")
        print(f"  {'Exact match':<25s} {base_em:>9.2f}% {ft_em:>11.2f}% {delta_em:>+9.2f}%")
        print(f"  {'No match':<25s} {base_nm:>9.2f}% {ft_nm:>11.2f}% {delta_nm:>+9.2f}%")
    else:
        ft_acc = all_results["finetuned"]["vqa_accuracy"]
        ft_em = all_results["finetuned"]["exact_match_pct"]
        print(f"  {'VQA Accuracy':<25s} {'(skipped)':>10s} {ft_acc:>11.2f}%")
        print(f"  {'Exact match':<25s} {'(skipped)':>10s} {ft_em:>11.2f}%")

    print("=" * 60)

    # ---- Save results ----
    output = {
        "dataset": VAL_DATASET_NAME,
        "split": VAL_SPLIT,
        "held_out_ratio": 1 - VAL_EVAL_RATIO,
        "split_seed": VAL_SPLIT_SEED,
        "model_dir": args.model_dir,
        "num_examples": len(dataset),
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
    if "base" in all_results:
        print("\nSample head-to-head comparisons:")
        base_examples = all_results["base"]["per_example"]
        ft_examples = all_results["finetuned"]["per_example"]

        for j in range(min(10, len(base_examples))):
            b = base_examples[j]
            f_ = ft_examples[j]
            print(f"  Q: {b['question'][:60]}...")
            print(f"    Base:       {b['predicted']:<20s} (score: {b['score']:.2f})")
            print(f"    Fine-tuned: {f_['predicted']:<20s} (score: {f_['score']:.2f})")
            print(f"    GT:         {b['ground_truth'][:3]}")
            print()


if __name__ == "__main__":
    main()