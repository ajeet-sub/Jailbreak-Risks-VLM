"""
MM-SafetyBench Evaluation on LLaVA v1.5-7b (with LoRA adapter support)

Loads the MM-SafetyBench dataset from HuggingFace, runs each prompt
through LLaVA v1.5-7b (optionally with one or more LoRA adapters),
and saves raw model responses to separate CSVs per adapter.

Requirements:
    pip install datasets transformers torch pillow pandas tqdm accelerate bitsandbytes peft

Usage:
    # Base model only
    python MM_SafetyBenchEval.py

    # Single LoRA adapter
    python MM_SafetyBenchEval.py --lora_paths /path/to/okvqa_lora --lora_names okvqa

    # Multiple LoRA adapters (evaluated sequentially, separate output CSVs)
    python MM_SafetyBenchEval.py \
        --lora_paths /path/to/okvqa_lora /path/to/gmai_lora \
        --lora_names okvqa gmai

    # With scenario/split control
    python MM_SafetyBenchEval.py \
        --lora_paths /path/to/okvqa_lora /path/to/gmai_lora \
        --lora_names okvqa gmai \
        --scenarios all --splits SD --max_eval 10
"""

import argparse
import os
import io
import time
import copy
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


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

def load_base_model(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    """Load the base LLaVA v1.5-7b model and processor."""
    print(f"Loading base model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"  Base model loaded on: {model.device}")
    return model, processor


def load_lora_adapter(base_model, lora_path: str, adapter_name: str):
    """
    Load a PEFT LoRA adapter on top of the base model.
    Returns the PeftModel wrapping the base model.
    """
    print(f"  Loading LoRA adapter '{adapter_name}' from: {lora_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        adapter_name=adapter_name,
    )
    peft_model.eval()
    print(f"  LoRA adapter '{adapter_name}' loaded successfully.")
    return peft_model


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
# 4. Run inference for one model variant
# ──────────────────────────────────────────────

def run_eval_loop(
    model,
    processor,
    sample_df: pd.DataFrame,
    output_path: str,
    model_label: str,
    max_new_tokens: int,
    checkpoint_every: int,
):
    """
    Run inference on all examples for a single model variant (base or LoRA).
    Saves results to output_path with checkpoint/resume support.
    """
    checkpoint_path = output_path + ".checkpoint.csv"
    start_idx = 0

    if os.path.exists(checkpoint_path):
        print(f"  Found checkpoint at {checkpoint_path}, resuming...")
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_df)
        results = checkpoint_df.to_dict("records")
        print(f"    Resuming from row {start_idx}")
    else:
        results = []

    print(f"\n  Running inference [{model_label}] on {len(sample_df)} examples "
          f"(starting from {start_idx})...")
    start_time = time.time()

    for idx in tqdm(range(start_idx, len(sample_df)), initial=start_idx,
                    total=len(sample_df), desc=f"  {model_label}"):
        row = sample_df.iloc[idx]

        image = load_image_from_row(row)
        prompt = row["question"]

        try:
            response = run_inference(model, processor, image, prompt, max_new_tokens)
        except Exception as e:
            response = f"[ERROR] {str(e)}"
            print(f"\n    Error on row {idx}: {e}")

        results.append({
            "row_id": idx,
            "model": model_label,
            "scenario": row["scenario"],
            "split": row["split"],
            "id": row["id"],
            "question": prompt,
            "response": response,
        })

        # Periodic checkpoint
        if (idx + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - start_time
    n = len(results_df)
    print(f"\n  Done [{model_label}]. {n} results saved to {output_path}")
    print(f"  Time: {elapsed / 60:.1f} minutes ({elapsed / max(1, n):.2f}s per sample)")
    print(f"  Errors: {results_df['response'].str.startswith('[ERROR]').sum()}")

    return results_df


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MM-SafetyBench prompts on LLaVA v1.5-7b with optional LoRA adapters"
    )
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Base LLaVA model ID on HuggingFace Hub")
    parser.add_argument("--dataset_id", type=str, default="PKU-Alignment/MM-SafetyBench",
                        help="MM-SafetyBench dataset ID")
    parser.add_argument("--lora_paths", type=str, nargs="*", default=[],
                        help="Paths to LoRA adapter directories (safetensors + adapter_config.json)")
    parser.add_argument("--lora_names", type=str, nargs="*", default=[],
                        help="Short names for each LoRA adapter (used in output filenames)")
    parser.add_argument("--out_dir", type=str, default="results_mm_safetybench",
                        help="Output directory for CSV results")
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

    # Validate LoRA args
    if args.lora_paths and not args.lora_names:
        # Auto-name from directory names
        args.lora_names = [os.path.basename(p.rstrip("/")) for p in args.lora_paths]
    if args.lora_paths and len(args.lora_paths) != len(args.lora_names):
        parser.error("--lora_paths and --lora_names must have the same number of entries")

    scenarios = ALL_SCENARIOS if "all" in args.scenarios else args.scenarios
    splits = args.splits

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the list of model variants to run
    # Each entry: (label, lora_path_or_None)
    variants = []
    if not args.lora_paths:
        # No LoRA specified → just run the base model
        variants.append(("base", None))
    else:
        for name, path in zip(args.lora_names, args.lora_paths):
            variants.append((name, path))

    print(f"\n{'='*60}")
    print("CONFIG")
    print(f"{'='*60}")
    print(f"  Base model:  {args.model_id}")
    print(f"  Dataset:     {args.dataset_id}")
    print(f"  Scenarios:   {scenarios}")
    print(f"  Splits:      {splits}")
    print(f"  Max eval:    {args.max_eval if args.max_eval else 'unlimited'}")
    print(f"  Variants:    {[v[0] for v in variants]}")
    print(f"  Output dir:  {args.out_dir}")
    print()

    # ── Load dataset (once, shared across all variants) ──
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

    # ── Load base model (once) ──
    base_model, processor = load_base_model(args.model_id)

    # ── Run each variant ──
    all_summaries = []

    for variant_name, lora_path in variants:
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {variant_name}")
        print(f"{'='*60}")

        if lora_path is not None:
            model = load_lora_adapter(base_model, lora_path, variant_name)
        else:
            model = base_model

        output_path = os.path.join(args.out_dir, f"{variant_name}_responses.csv")

        results_df = run_eval_loop(
            model=model,
            processor=processor,
            sample_df=sample_df,
            output_path=output_path,
            model_label=variant_name,
            max_new_tokens=args.max_new_tokens,
            checkpoint_every=args.checkpoint_every,
        )

        all_summaries.append({
            "variant": variant_name,
            "total": len(results_df),
            "errors": int(results_df["response"].str.startswith("[ERROR]").sum()),
            "output": output_path,
        })

        # Properly unload LoRA adapter before next variant
        # PeftModel.from_pretrained wraps base_model in-place, so we must
        # unwrap it to avoid stacking adapters on subsequent iterations.
        if lora_path is not None:
            base_model = model.unload()
            del model
            torch.cuda.empty_cache()

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("ALL VARIANTS COMPLETE")
    print(f"{'='*60}")
    for s in all_summaries:
        print(f"  {s['variant']:<20} responses={s['total']}  errors={s['errors']}  → {s['output']}")


if __name__ == "__main__":
    main()