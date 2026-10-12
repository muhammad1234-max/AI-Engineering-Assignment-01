"""Generate F08: Authentic autoregressive text samples from genuine trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.checkpoint import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROMPTS = [
    "ROMEO:\n",
    "JULIET:\n",
    "HAMLET:\n",
    "To be, or not to be",
    "O, swear not by the moon",
]


def generate_samples_evidence(
    checkpoint_path: Path | None = None,
    *,
    temperature: float = 0.8,
    top_k: int = 40,
    max_new_tokens: int = 80,
    seed: int = 42,
) -> Path:
    """Generate completions from 5 fixed prompts and create F08 visual evidence."""
    if checkpoint_path is None:
        checkpoint_path = ROOT / "checkpoints/best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model before generating samples."
        )

    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    model.eval()

    torch.manual_seed(seed)
    results = []

    samples_dir = ROOT / "outputs/samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_text_file = samples_dir / "generated_samples.txt"
    text_output_lines = []

    for idx, prompt in enumerate(PROMPTS, 1):
        prompt_ids = tokenizer.encode(prompt, as_tensor=True).unsqueeze(0)
        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        output_text = tokenizer.decode(gen_ids[0])
        results.append((prompt, output_text))

        text_output_lines.append(f"=== Sample {idx} (Prompt: {repr(prompt)}) ===")
        text_output_lines.append(output_text)
        text_output_lines.append("\n" + "-" * 50 + "\n")

    samples_text_file.write_text("\n".join(text_output_lines), encoding="utf-8")

    # Render clean visual card for F08
    fig, axes = plt.subplots(len(PROMPTS), 1, figsize=(12, 10), dpi=140)

    for i, (prompt, text) in enumerate(results):
        ax = axes[i]
        ax.axis("off")
        card_text = (
            f"Prompt {i + 1}: {repr(prompt.strip())}  |  Settings: T={temperature}, top_k={top_k}, seed={seed}\n"
            f"{'─' * 70}\n"
            f"{text.strip()}"
        )
        ax.text(
            0.02,
            0.85,
            card_text,
            transform=ax.transAxes,
            fontsize=8.5,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da"),
        )

    fig.suptitle(
        "F08 — Genuine Final Autoregressive Generated Samples (5 Fixed Prompts)\n"
        "Trained Tiny Shakespeare GPT Model | No External Post-Editing",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    out_path = ROOT / "outputs/figures/F08_samples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    dest = save_evidence(
        out_path,
        sources=[
            "experiments/generate_samples.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/gpt.py",
        ],
        details={
            "command": "python -m experiments.generate_samples",
            "kind": "authentic generated samples from 5 fixed prompts",
            "checkpoint": str(checkpoint_path.name),
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
            "prompts": PROMPTS,
            "text_output": "outputs/samples/generated_samples.txt",
        },
    )
    print(f"Saved F08 evidence to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Generate F08 authentic samples")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()
    generate_samples_evidence(
        args.checkpoint,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
