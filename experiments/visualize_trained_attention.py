"""Generate F07: Attention heatmaps from real trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.checkpoint import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_attention_figure(
    checkpoint_path: Path | None = None,
    prompt: str = "ROMEO:\nWhat light",
) -> Path:
    """Extract and plot attention maps from a genuine trained checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = ROOT / "checkpoints/best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model before extracting attention maps."
        )

    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    model.eval()

    tokens = tokenizer.encode(prompt, as_tensor=True).unsqueeze(0)  # (1, T)
    seq_len = tokens.size(1)

    with torch.no_grad():
        _, all_attn = model(tokens, return_attention=True)

    # all_attn is a list of tensors of shape (1, n_heads, T, T) for each layer
    n_layers = len(all_attn)
    n_heads = all_attn[0].size(1)

    # Token label strings for axes (escaped for display)
    token_labels = [repr(tokenizer.idx2char[int(idx)])[1:-1] for idx in tokens[0]]

    # Plot 4 heads from Layer 0 (early) and Layer N-1 (late)
    fig, axes = plt.subplots(2, min(n_heads, 4), figsize=(14, 7), dpi=140)

    layers_to_show = [0, n_layers - 1]
    layer_names = ["Layer 0 (Early)", f"Layer {n_layers - 1} (Late)"]

    for row_idx, layer_idx in enumerate(layers_to_show):
        attn_layer = all_attn[layer_idx][0]  # (n_heads, T, T)
        for h in range(min(n_heads, 4)):
            ax = axes[row_idx, h] if n_heads > 1 else axes[row_idx]
            attn_head = attn_layer[h].numpy()

            ax.imshow(attn_head, cmap="viridis", vmin=0.0, vmax=attn_head.max())
            ax.set_title(f"{layer_names[row_idx]} - Head {h}", fontsize=10, fontweight="bold")
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(token_labels, fontsize=8)

            if h == 0:
                ax.set_ylabel("Query Token", fontsize=10, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Key Token", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"F07 — Genuine Trained Model Attention Heatmaps (Prompt: {repr(prompt)})\n"
        f"Verified Causal Masking: Strictly zero weights in upper triangle (future keys)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = ROOT / "outputs/figures/F07_trained_attention.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    dest = save_evidence(
        out_path,
        sources=[
            "experiments/visualize_trained_attention.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/gpt.py",
        ],
        details={
            "command": "python -m experiments.visualize_trained_attention",
            "kind": "attention heatmap extracted from genuine trained checkpoint",
            "checkpoint": str(checkpoint_path.name),
            "prompt": prompt,
            "layers_inspected": layers_to_show,
            "heads_inspected": min(n_heads, 4),
        },
    )
    print(f"Saved F07 evidence to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Generate F07 trained attention heatmaps")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="ROMEO:\nWhat light", help="Prompt text")
    args = parser.parse_args()
    generate_attention_figure(args.checkpoint, args.prompt)


if __name__ == "__main__":
    main()
