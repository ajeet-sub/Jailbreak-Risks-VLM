"""
MM-SafetyBench Evaluation on LLaVA v1.5-7b

Loads the MM-SafetyBench dataset from HuggingFace, runs each prompt
through LLaVA v1.5-7b, and saves raw model responses to a CSV.

Requirements:
    pip install datasets transformers torch pillow pandas tqdm accelerate bitsandbytes

Usage:
    python MM_SafetyBenchEval.py [--output results.csv] [--scenarios all] [--splits SD] [--checkpoint_every 50]
"""

import argparse
import os
import io
import time
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


# ──────────────────────────────────────────────
# MM-SafetyBench scenarios and splits
# ──────────────────────────────────────────────

ALL_SCENARIOS = [
    "EconomicHarm",
    "Financial_Advice",
    "Fraud",
    "Gov_Decision",
    "HateSpeech",
    "Health_Consultation",
    "Illegal_Activitiy",   # note: original typo in dataset name
    "Legal_Opinion",
    "Malware_Generation",
    "Physical_Harm",
    "Political_Lobbying",
    "Privacy_Violence",
    "Sex",
]

ALL_SPLITS = ["SD", "SD_TYPO", "TYPO", "Text_only"]


# ──────────────────────────────────────────────
# 1. Dataset loading
# ──────────────────────────────────────────────

def load_scenario_data(dataset_id: str, scenarios: list, splits: list) -> pd.DataFrame:
    """
    Load MM-SafetyBench scenarios/splits from HuggingFace and flatten
    into a single DataFrame with columns: scenario, split, id, question, image.
    """
    print("Loading MM-SafetyBench dataset from HuggingFace...")
    all_rows = []

    for scenario in scenarios:
        print(f"  Loading scenario: {scenario}")
        try:
            dsd = load_dataset(dataset_id, scenario, trust_remote_code=True)
        except Exception as e:
            print(f"    [WARN] Could not load scenario '{scenario}': {e}")
            continue

        available_splits = list(dsd.keys())

        for split in splits:
            if split not in available_splits:
                print(f"    [WARN] Split '{split}' not in {available_splits}; skipping.")
                continue

            ds = dsd[split]
            print(f"    Split '{split}': {len(ds)} examples")

            for i in range(len(ds)):
                ex = ds[i]
                question = _pick_str(ex.get("question"), ex.get("prompt"), ex.get("text"))
                if not question:
                    continue
                all_rows.append({
                    "scenario": scenario,
                    "split": split,
                    "id": ex.get("id", i),
                    "question": question,
                    "image": ex.get("image"),
                })

    df = pd.DataFrame(all_rows)
    print(f"\n  Total examples loaded: {len(df)}")
    print(f"  Distribution by scenario:\n{df['scenario'].value_counts().to_string()}")
    print(f"  Distribution by split:\n{df['split'].value_counts().to_string()}")
    return df


def _pick_str(*candidates) -> str | None:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


# ──────────────────────────────────────────────
# 2. Model loading
# ──────────────────────────────────────────────

def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """Load LLaVA v1.5-7b and its processor from HuggingFace."""
    print(f"Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"  Model loaded on: {model.device}")
    return model, processor


# ──────────────────────────────────────────────
# 3. Inference
# ──────────────────────────────────────────────

def run_inference(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
    """Run a single (image, prompt) pair through LLaVA and return the response."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated portion
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def load_image_from_row(row) -> Image.Image:
    """
    Extract the image from a dataset row. MM-SafetyBench images arrive as
    PIL Image objects, dicts with a 'bytes' key, or raw bytes. For the
    'Text_only' split there is no image, so fall back to a blank image.
    """
    if row["split"] == "Text_only":
        return Image.new("RGB", (336, 336), (255, 255, 255))

    raw = row.get("image")
    if raw is not None:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, dict):
            b = raw.get("bytes") or raw.get("data")
            if b:
                return Image.open(io.BytesIO(b)).convert("RGB")
            p = raw.get("path")
            if p:
                return Image.open(p).convert("RGB")

    # Fallback: blank image
    print(f"  Warning: No image data for row, using blank image.")
    return Image.new("RGB", (336, 336), (255, 255, 255))


# ──────────────────────────────────────────────
# 4. Main loop
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MM-SafetyBench prompts on LLaVA v1.5-7b"
    )
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model ID on HuggingFace Hub")
    parser.add_argument("--dataset_id", type=str, default="PKU-Alignment/MM-SafetyBench",
                        help="MM-SafetyBench dataset ID")
    parser.add_argument("--output", type=str, default="mm_safetybench_results.csv",
                        help="Output CSV path")
    parser.add_argument("--scenarios", type=str, nargs="+", default=["all"],
                        choices=["all"] + ALL_SCENARIOS,
                        help="Which scenarios to evaluate (default: all)")
    parser.add_argument("--splits", type=str, nargs="+", default=["SD"],
                        choices=ALL_SPLITS,
                        help="Which splits to evaluate (default: SD)")
    parser.add_argument("--max_eval", type=int, default=0,
                        help="Max examples per scenario × split (0 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per response")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every N rows")
    args = parser.parse_args()

    scenarios = ALL_SCENARIOS if "all" in args.scenarios else args.scenarios
    splits = args.splits

    print(f"\n{'='*60}")
    print("CONFIG")
    print(f"{'='*60}")
    print(f"  Model:      {args.model_id}")
    print(f"  Dataset:    {args.dataset_id}")
    print(f"  Scenarios:  {scenarios}")
    print(f"  Splits:     {splits}")
    print(f"  Max eval:   {args.max_eval if args.max_eval else 'unlimited'}")
    print(f"  Output:     {args.output}")
    print()

    # ── Load dataset ──
    sample_df = load_scenario_data(args.dataset_id, scenarios, splits)

    if sample_df.empty:
        print("[ERROR] No examples loaded. Check dataset/model access.")
        return

    # Apply max_eval limit per scenario × split
    if args.max_eval and args.max_eval > 0:
        sample_df = (
            sample_df
            .groupby(["scenario", "split"], group_keys=False)
            .head(args.max_eval)
            .reset_index(drop=True)
        )
        print(f"  After max_eval cap: {len(sample_df)} examples")

    # ── Load model ──
    model, processor = load_model(args.model_id)

    # ── Resume from checkpoint if exists ──
    checkpoint_path = args.output + ".checkpoint.csv"
    start_idx = 0

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming...")
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_df)
        results = checkpoint_df.to_dict("records")
        print(f"  Resuming from row {start_idx}")
    else:
        results = []

    # ── Run inference ──
    print(f"\nRunning inference on {len(sample_df)} examples (starting from {start_idx})...")
    start_time = time.time()

    for idx in tqdm(range(start_idx, len(sample_df)), initial=start_idx, total=len(sample_df)):
        row = sample_df.iloc[idx]

        image = load_image_from_row(row)
        prompt = row["question"]

        try:
            response = run_inference(model, processor, image, prompt, args.max_new_tokens)
        except Exception as e:
            response = f"[ERROR] {str(e)}"
            print(f"\n  Error on row {idx}: {e}")

        results.append({
            "row_id": idx,
            "scenario": row["scenario"],
            "split": row["split"],
            "id": row["id"],
            "question": prompt,
            "response": response,
        })

        # Periodic checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    # ── Save final results ──
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\nDone. {len(results_df)} results saved to {args.output}")
    print(f"Total time: {elapsed / 60:.1f} minutes ({elapsed / len(results_df):.2f}s per sample)")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total responses: {len(results_df)}")
    print(f"  Errors: {results_df['response'].str.startswith('[ERROR]').sum()}")
    print(f"\nBy scenario:\n{results_df['scenario'].value_counts().to_string()}")
    print(f"\nBy split:\n{results_df['split'].value_counts().to_string()}")


if __name__ == "__main__":
    main()