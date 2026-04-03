"""
JailBreakV-28K LLM Transfer Attack Evaluation on LLaVA v1.5-7b

Loads the JailBreakV-28K dataset from HuggingFace, samples 7000 LLM transfer
attack prompts (stratified by policy category and attack type), runs them
through LLaVA v1.5-7b, and saves results to a CSV.

Requirements:
    pip install datasets transformers torch pillow pandas tqdm accelerate bitsandbytes

Usage:
    python run_jailbreakv_llm_transfer.py [--output results.csv] [--seed 42] [--batch_checkpoint 50]
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
# 1. Dataset loading and sampling
# ──────────────────────────────────────────────

def load_and_sample(seed: int, n_samples: int = 7000) -> pd.DataFrame:
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
    per_stratum = n_samples // n_strata  # base allocation
    remainder = n_samples % n_strata

    sampled_frames = []
    groups = list(df_transfer.groupby(["policy", "format"]))

    # Sort groups for deterministic remainder assignment
    groups.sort(key=lambda x: x[0])

    for i, ((policy, fmt), group_df) in enumerate(groups):
        # Distribute remainder across first few strata
        take = per_stratum + (1 if i < remainder else 0)
        take = min(take, len(group_df))  # can't take more than available
        sampled = group_df.sample(n=take, random_state=seed)
        sampled_frames.append(sampled)

    sample_df = pd.concat(sampled_frames, ignore_index=True)

    # If we're short (some strata too small), top up from remaining rows
    if len(sample_df) < n_samples:
        already_sampled = set(sample_df.index)
        remaining_pool = df_transfer[~df_transfer.index.isin(already_sampled)]
        shortfall = n_samples - len(sample_df)
        top_up = remaining_pool.sample(n=min(shortfall, len(remaining_pool)), random_state=seed)
        sample_df = pd.concat([sample_df, top_up], ignore_index=True)

    # Shuffle the final sample
    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  Final sample size: {len(sample_df)}")
    print(f"  Sample distribution by policy:\n{sample_df['policy'].value_counts().to_string()}")
    print(f"  Sample distribution by attack type:\n{sample_df['format'].value_counts().to_string()}")

    return sample_df


# ──────────────────────────────────────────────
# 2. Model loading
# ──────────────────────────────────────────────

def load_model(device: str = "cuda"):
    """Load LLaVA v1.5-7b and its processor from HuggingFace."""
    model_id = "llava-hf/llava-1.5-7b-hf"
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


def load_image_from_dataset_row(row) -> Image.Image:
    """
    Extract the image from a dataset row. The HuggingFace dataset stores
    images as PIL Image objects or bytes in the image column. If that fails,
    fall back to a blank 336x336 white image.
    """
    # The dataset may have an 'image' column with actual image data,
    # or we may need to use image_path. Try both approaches.
    if "image" in row and row["image"] is not None:
        img = row["image"]
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert("RGB")

    # Fallback: blank image (matches LLaVA's expected input size)
    print(f"  Warning: No image data for row, using blank image.")
    return Image.new("RGB", (336, 336), (255, 255, 255))


# ──────────────────────────────────────────────
# 4. Main loop
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run JailBreakV-28K LLM transfer attacks on LLaVA v1.5-7b"
    )
    parser.add_argument("--output", type=str, default="jailbreakv_llm_transfer_results.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--n_samples", type=int, default=7000,
                        help="Number of samples to run")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per response")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every N rows")
    args = parser.parse_args()

    # ── Load dataset and sample ──
    sample_df = load_and_sample(seed=args.seed, n_samples=args.n_samples)

    # ── Load model ──
    model, processor = load_model()

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
    print(f"\nRunning inference on {len(sample_df)} samples (starting from {start_idx})...")
    start_time = time.time()

    for idx in tqdm(range(start_idx, len(sample_df)), initial=start_idx, total=len(sample_df)):
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
            "policy": row["policy"],
            "format": row["format"],
            "jailbreak_query": prompt,
            "redteam_query": row["redteam_query"],
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

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(results_df)}")
    print(f"Errors: {results_df['response'].str.startswith('[ERROR]').sum()}")
    print(f"\nBy policy:\n{results_df['policy'].value_counts().to_string()}")
    print(f"\nBy attack type:\n{results_df['format'].value_counts().to_string()}")


if __name__ == "__main__":
    main()