"""Command-line interface for interactive and one-shot text generation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from zaid_lm.checkpoint import load_checkpoint


def generate_text(
    checkpoint_path: Path | str,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    device: str = "cpu",
) -> dict:
    """Generate text from a trained model checkpoint."""
    if seed is not None:
        torch.manual_seed(seed)

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    model.eval()

    # Unescape common sequences like \n from terminal arguments
    prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

    # Filter out characters not in vocabulary to avoid crash on CLI input
    filtered_chars = [ch for ch in prompt if ch in tokenizer.char2idx]
    if not filtered_chars:
        prompt = "\n"  # fallback default token
    else:
        prompt = "".join(filtered_chars)

    prompt_ids = tokenizer.encode(prompt, as_tensor=True).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    elapsed = time.time() - start_time

    generated_text = tokenizer.decode(out_ids[0].cpu())
    new_tokens_count = out_ids.size(1) - prompt_ids.size(1)
    tok_per_sec = new_tokens_count / elapsed if elapsed > 0 else 0.0

    return {
        "prompt": prompt,
        "completion": generated_text[len(prompt) :],
        "full_text": generated_text,
        "new_tokens": new_tokens_count,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tok_per_sec,
        "model_step": ckpt["step"],
        "val_loss": ckpt["val_loss"],
    }


def main():
    parser = argparse.ArgumentParser(description="Shakespearean LLM — Text Generation CLI")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Conditioning text prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum new tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=None, help="Random number generator seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (cpu or cuda)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    # Interactive mode if prompt not provided via CLI
    if args.prompt is None:
        print("=" * 65)
        print("🎭 Shakespearean LLM Interactive Generation Console (Tiny Shakespeare)")
        print(f"Loaded Checkpoint: {args.checkpoint}")
        print("Type your prompt and press Enter. (Type 'exit' to quit)")
        print("=" * 65)

        while True:
            try:
                prompt = input("\nEnter Prompt > ")
                if prompt.strip().lower() in ("exit", "quit"):
                    break
                if not prompt:
                    continue

                res = generate_text(
                    args.checkpoint,
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed,
                    device=args.device,
                )
                print("\n--- Output ---")
                print(res["full_text"])
                print(
                    f"\n[Generated {res['new_tokens']} tokens in {res['elapsed_seconds']:.2f}s "
                    f"({res['tokens_per_second']:.1f} tok/s)]"
                )
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        res = generate_text(
            args.checkpoint,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            device=args.device,
        )
        print("=" * 65)
        print(f"Prompt: {repr(args.prompt)}")
        print(f"Generated text:\n{res['full_text']}")
        print(
            f"\n[Generated {res['new_tokens']} tokens in {res['elapsed_seconds']:.2f}s "
            f"({res['tokens_per_second']:.1f} tok/s)]"
        )
        print("=" * 65)


if __name__ == "__main__":
    main()
