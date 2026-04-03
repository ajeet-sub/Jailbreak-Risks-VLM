"""
Generation sanity check for fine-tuned models.

Tests whether the model still produces full natural language
responses on open-ended prompts, or whether fine-tuning on
short-answer MCQ data has collapsed its generation distribution.

Loads random OK-VQA images and prompts the model with open-ended
questions that require multi-word/multi-sentence responses.
If the model outputs single letters or truncated nonsense,
the fine-tune has a generation collapse problem.

Usage:
    python sanity_check_generation.py --model_dir ./llava-gmai-finetuned
    python sanity_check_generation.py --model_dir ./llava-okvqa-finetuned
    python sanity_check_generation.py --base_only  # test base model alone
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from PIL import Image
import argparse
import random

BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_DATASET = "lmms-lab/OK-VQA"
IMAGE_SPLIT = "val2014"
NUM_IMAGES = 5
SEED = 42

OPEN_ENDED_PROMPTS = [
    "Describe what you see in this image in detail.",
    "What is happening in this photograph?",
    "Explain everything visible in this image as thoroughly as you can.",
    "What objects, people, or activities can you identify in this image?",
    "If you were describing this image to someone who cannot see it, what would you say?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generation sanity check")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (omit or use --base_only for base model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL_ID,
        help="Base model ID",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Only test the base model (no adapter)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=NUM_IMAGES,
        help="Number of random images to test",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Max tokens to generate per response",
    )
    return parser.parse_args()


def test_model(model, processor, images, model_name, max_new_tokens):
    """Run open-ended prompts on random images and print responses."""

    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print(f"{'='*70}")

    model.eval()

    total_tokens = 0
    single_letter_count = 0
    short_response_count = 0  # < 10 words

    for i, image in enumerate(images):
        prompt_text = OPEN_ENDED_PROMPTS[i % len(OPEN_ENDED_PROMPTS)]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_text = processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text.strip()

        word_count = len(response.split())
        total_tokens += word_count

        if len(response) <= 2:
            single_letter_count += 1
        if word_count < 10:
            short_response_count += 1

        print(f"\n  Image {i+1} | Prompt: {prompt_text}")
        print(f"  Response ({word_count} words):")
        print(f"  >>> {response[:500]}")
        if len(response) > 500:
            print(f"  ... [{len(response) - 500} more characters]")

    # Summary
    avg_words = total_tokens / len(images)
    print(f"\n{'-'*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'-'*70}")
    print(f"  Images tested:        {len(images)}")
    print(f"  Avg words/response:   {avg_words:.1f}")
    print(f"  Single-letter only:   {single_letter_count}/{len(images)}")
    print(f"  Short (<10 words):    {short_response_count}/{len(images)}")

    if single_letter_count > len(images) // 2:
        print(f"\n  *** GENERATION COLLAPSE DETECTED ***")
        print(f"  The model is producing single-letter outputs on open-ended")
        print(f"  prompts. This means fine-tuning has shifted the generation")
        print(f"  distribution. Adversarial evaluation results would be")
        print(f"  uninterpretable — retrain with a format instruction.")
    elif short_response_count > len(images) // 2:
        print(f"\n  *** WARNING: SHORT RESPONSES ***")
        print(f"  The model is producing unusually short responses. This may")
        print(f"  indicate partial generation collapse. Check the outputs above.")
    else:
        print(f"\n  PASS: Model generates full natural language responses.")

    return {
        "model_name": model_name,
        "avg_words": avg_words,
        "single_letter_count": single_letter_count,
        "short_response_count": short_response_count,
    }


def main():
    args = parse_args()

    # Load random images
    print(f"Loading {args.num_images} random images from {IMAGE_DATASET}...")
    dataset = load_dataset(IMAGE_DATASET, split=IMAGE_SPLIT)

    random.seed(SEED)
    indices = random.sample(range(len(dataset)), args.num_images)

    images = []
    for idx in indices:
        img = dataset[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    print(f"Loaded {len(images)} images (indices: {indices})")

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "right"

    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Always test base model first as reference
    base_results = test_model(
        base_model, processor, images,
        model_name="LLaVA-1.5-7B (base)",
        max_new_tokens=args.max_new_tokens,
    )

    # Test fine-tuned model if provided
    if not args.base_only and args.model_dir:
        print(f"\nLoading LoRA adapter from: {args.model_dir}")
        finetuned_model = PeftModel.from_pretrained(base_model, args.model_dir)

        ft_results = test_model(
            finetuned_model, processor, images,
            model_name=f"LLaVA-1.5-7B + LoRA ({args.model_dir})",
            max_new_tokens=args.max_new_tokens,
        )

        # Side-by-side comparison
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Metric':<30s} {'Base':>12s} {'Fine-tuned':>12s}")
        print(f"  {'-'*30} {'-'*12} {'-'*12}")
        print(f"  {'Avg words/response':<30s} {base_results['avg_words']:>11.1f}  {ft_results['avg_words']:>11.1f}")
        print(f"  {'Single-letter outputs':<30s} {base_results['single_letter_count']:>12d} {ft_results['single_letter_count']:>12d}")
        print(f"  {'Short responses (<10w)':<30s} {base_results['short_response_count']:>12d} {ft_results['short_response_count']:>12d}")
        print(f"{'='*70}")

        if ft_results["avg_words"] < base_results["avg_words"] * 0.3:
            print("\n  *** The fine-tuned model's avg response length is <30% of")
            print("  the base model's. This suggests generation collapse. ***")


if __name__ == "__main__":
    main()