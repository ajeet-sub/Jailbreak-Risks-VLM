"""
JailBreakV-28K LLM Transfer Attack Evaluation on LoRA Fine-Tuned LLaVA v1.5-7b

Loads the JailBreakV-28K dataset from HuggingFace, samples LLM transfer
attack prompts (stratified by policy category and attack type), runs them
through a LoRA fine-tuned LLaVA v1.5-7b, and saves results to a CSV.

Supports running the base model (no adapter) or any number of LoRA adapters,
making it easy to compare base vs fine-tuned safety regression.

Requirements:
    pip install datasets transformers torch pillow pandas tqdm accelerate peft bitsandbytes

Usage:
    # Run on base model (no LoRA)
    python run_jailbreakv_finetuned.py --output results_base.csv

    # Run on a single LoRA fine-tuned model
    python run_jailbreakv_finetuned.py \
        --lora_path /path/to/lora/adapter \
        --model_name my_finetuned_model \
        --output results_finetuned.csv

    # Run on multiple LoRA adapters in sequence
    python run_jailbreakv_finetuned.py \
        --lora_path /path/to/adapter1 /path/to/adapter2 \
        --model_name adapter1 adapter2 \
        --output results_multi.csv
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
from peft import PeftModel


# ──────────────────────────────────────────────
# 1. Dataset loading and sampling
# ──────────────────────────────────────────────

def load_and_sample(seed: int, n_samples: int = 12000) -> pd.DataFrame:
    """
    Load JailBreakV-28K from HuggingFace, filter to LLM transfer attacks,
    and return a stratified sample of n_samples rows balanced across
    (policy, format) combinations.
    """
    print("Loading JailBreakV-28K dataset from HuggingFace...")
    ds = load_dataset(
        "JailbreakV-28K/JailBreakV-28k",
        "JailBreakV_28K",
        split="JailBreakV_28K",
    )
    df = ds.to_pandas()
    print(f"  Full dataset: {len(df)} rows")

    # Filter to LLM transfer attacks only
    df_transfer = df[df["transfer_from_llm"] == True].copy()
    df_transfer = df_transfer.reset_index(drop=True)
    print(f"  LLM transfer subset: {len(df_transfer)} rows")

    # Show distribution before sampling
    strata = df_transfer.groupby(["policy", "format"]).size().reset_index(name="count")
    print(f"  Unique (policy, format) strata: {len(strata)}")
    print(f"  Policy categories: {df_transfer['policy'].nunique()}")
    print(f"  Attack types: {df_transfer['format'].unique().tolist()}")

    # Stratified sampling: equal draw from each (policy, format) group
    n_strata = len(strata)
    per_stratum = n_samples // n_strata
    remainder = n_samples % n_strata

    sampled_frames = []
    groups = list(df_transfer.groupby(["policy", "format"]))
    groups.sort(key=lambda x: x[0])

    for i, ((policy, fmt), group_df) in enumerate(groups):
        take = per_stratum + (1 if i < remainder else 0)
        take = min(take, len(group_df))
        sampled = group_df.sample(n=take, random_state=seed)
        sampled_frames.append(sampled)

    sample_df = pd.concat(sampled_frames, ignore_index=True)

    # Top up if short
    if len(sample_df) < n_samples:
        already_sampled = set(sample_df.index)
        remaining_pool = df_transfer[~df_transfer.index.isin(already_sampled)]
        shortfall = n_samples - len(sample_df)
        top_up = remaining_pool.sample(n=min(shortfall, len(remaining_pool)), random_state=seed)
        sample_df = pd.concat([sample_df, top_up], ignore_index=True)

    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  Final sample size: {len(sample_df)}")
    print(f"  Sample distribution by policy:\n{sample_df['policy'].value_counts().to_string()}")
    print(f"  Sample distribution by attack type:\n{sample_df['format'].value_counts().to_string()}")

    return sample_df


# ──────────────────────────────────────────────
# 2. Model loading
# ──────────────────────────────────────────────

def load_base_model(device: str = "cuda"):
    """Load the base LLaVA v1.5-7b model and processor."""
    model_id = "llava-hf/llava-1.5-7b-hf"
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


def load_lora_model(base_model, lora_path: str):
    """
    Apply a LoRA adapter to the base model.
    Returns the PEFT-wrapped model. The base model weights are shared,
    so swapping adapters is lightweight.
    """
    print(f"  Loading LoRA adapter from: {lora_path}")

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Print adapter info
    if hasattr(model, 'peft_config'):
        for name, config in model.peft_config.items():
            print(f"    Adapter '{name}': r={config.r}, "
                  f"lora_alpha={config.lora_alpha}, "
                  f"target_modules={config.target_modules}")

    return model


def unload_lora(model):
    """
    Remove the LoRA adapter and return the base model.
    Useful when cycling through multiple adapters.
    """
    if isinstance(model, PeftModel):
        print("  Unloading LoRA adapter...")
        model = model.unload()
    return model


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

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def load_image_from_dataset_row(row) -> Image.Image:
    """
    Extract the image from a dataset row. Falls back to a blank 336x336
    white image if no image data is available.
    """
    if "image" in row and row["image"] is not None:
        img = row["image"]
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert("RGB")

    print(f"  Warning: No image data for row, using blank image.")
    return Image.new("RGB", (336, 336), (255, 255, 255))


# ──────────────────────────────────────────────
# 4. Run evaluation for a single model config
# ──────────────────────────────────────────────

def run_evaluation(model, processor, sample_df, model_name, args):
    """
    Run inference on all samples for a single model configuration.
    Returns a DataFrame of results.
    """
    output_path = args.output.replace(".csv", f"_{model_name}.csv")
    checkpoint_path = output_path + ".checkpoint.csv"
    start_idx = 0

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"  Found checkpoint at {checkpoint_path}, resuming...")
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_df)
        results = checkpoint_df.to_dict("records")
        print(f"  Resuming from row {start_idx}")
    else:
        results = []

    print(f"\n  Running inference for '{model_name}' on {len(sample_df)} samples "
          f"(starting from {start_idx})...")
    start_time = time.time()

    for idx in tqdm(range(start_idx, len(sample_df)), initial=start_idx,
                    total=len(sample_df), desc=model_name):
        row = sample_df.iloc[idx]

        image = load_image_from_dataset_row(row)
        prompt = row["jailbreak_query"]

        try:
            response = run_inference(model, processor, image, prompt, args.max_new_tokens)
        except Exception as e:
            response = f"[ERROR] {str(e)}"
            print(f"\n  Error on row {idx}: {e}")

        results.append({
            "row_id": idx,
            "model_name": model_name,
            "policy": row["policy"],
            "format": row["format"],
            "jailbreak_query": prompt,
            "redteam_query": row["redteam_query"],
            "response": response,
        })

        if (idx + 1) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - start_time
    n_results = len(results_df) - start_idx
    print(f"\n  '{model_name}' done. {len(results_df)} results saved to {output_path}")
    if n_results > 0:
        print(f"  Time: {elapsed / 60:.1f} min ({elapsed / n_results:.2f}s per sample)")

    return results_df


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run JailBreakV-28K LLM transfer attacks on base or LoRA fine-tuned LLaVA v1.5-7b"
    )
    parser.add_argument("--output", type=str, default="jailbreakv_finetuned_results.csv",
                        help="Output CSV path (model name will be appended)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--n_samples", type=int, default=12000,
                        help="Number of samples to run")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per response")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every N rows")
    parser.add_argument("--lora_path", type=str, nargs="*", default=None,
                        help="Path(s) to LoRA adapter(s). If omitted, runs base model only.")
    parser.add_argument("--model_name", type=str, nargs="*", default=None,
                        help="Name(s) for each model config. Must match --lora_path count.")
    parser.add_argument("--include_base", action="store_true",
                        help="Also run the base model (useful when comparing base vs fine-tuned)")
    args = parser.parse_args()

    # Validate arguments
    if args.lora_path and args.model_name:
        if len(args.lora_path) != len(args.model_name):
            raise ValueError(
                f"--lora_path ({len(args.lora_path)}) and --model_name "
                f"({len(args.model_name)}) must have the same number of entries"
            )

    # ── Load dataset once (shared across all model configs) ──
    sample_df = load_and_sample(seed=args.seed, n_samples=args.n_samples)

    # ── Load base model ──
    base_model, processor = load_base_model()

    all_results = []

    # ── Run base model if requested ──
    if args.include_base or args.lora_path is None:
        name = "base"
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        results_df = run_evaluation(base_model, processor, sample_df, name, args)
        all_results.append(results_df)

    # ── Run each LoRA adapter ──
    if args.lora_path:
        for i, lora_path in enumerate(args.lora_path):
            name = args.model_name[i] if args.model_name else f"lora_{i}"

            print(f"\n{'='*60}")
            print(f"Evaluating: {name} (adapter: {lora_path})")
            print(f"{'='*60}")

            # Load LoRA on top of base
            lora_model = load_lora_model(base_model, lora_path)

            results_df = run_evaluation(lora_model, processor, sample_df, name, args)
            all_results.append(results_df)

            # Unload adapter to free memory before next one
            del lora_model
            torch.cuda.empty_cache()

    # ── Save combined results if multiple models ──
    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nCombined results ({len(combined)} rows) saved to {args.output}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for results_df in all_results:
        name = results_df["model_name"].iloc[0]
        total = len(results_df)
        errors = results_df["response"].str.startswith("[ERROR]").sum()
        print(f"\n  Model: {name}")
        print(f"  Total samples: {total}")
        print(f"  Errors: {errors}")
        print(f"  By policy:\n{results_df['policy'].value_counts().to_string()}")
        print(f"  By attack type:\n{results_df['format'].value_counts().to_string()}")


if __name__ == "__main__":
    main()