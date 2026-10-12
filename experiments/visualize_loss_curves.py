"""Generate F06: Genuine training and validation loss curves with perplexity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from experiments.phase3_evidence import ROOT, save_evidence

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_loss_figure(history_path: Path | None = None) -> Path:
    """Plot training and validation loss from actual logged history."""
    if history_path is None:
        history_path = ROOT / "outputs/training_history.json"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found at {history_path}. Please run training first."
        )

    data = json.loads(history_path.read_text(encoding="utf-8"))
    history = data.get("history", [])
    if not history:
        raise ValueError("No history entries found in training log.")

    steps = [entry["step"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]
    val_ppls = [entry["val_perplexity"] for entry in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)

    # 1. Loss curves
    ax1.plot(steps, train_losses, "b-o", label="Training Loss", linewidth=1.8, markersize=4)
    ax1.plot(steps, val_losses, "r-s", label="Validation Loss", linewidth=1.8, markersize=4)
    ax1.set_xlabel("Optimization Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11, fontweight="bold")
    ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    best_val = min(val_losses)
    best_step = steps[val_losses.index(best_val)]
    ax1.axvline(
        best_step, color="green", linestyle=":", alpha=0.8, label=f"Best Val: {best_val:.3f}"
    )
    ax1.annotate(
        f"Best: {best_val:.3f}\n(Step {best_step})",
        xy=(best_step, best_val),
        xytext=(best_step + 50, best_val + 0.3),
        arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="green"),
    )

    # 2. Perplexity curve
    ax2.plot(steps, val_ppls, "m-^", label="Validation Perplexity", linewidth=1.8, markersize=4)
    ax2.set_xlabel("Optimization Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Perplexity (exp(Loss))", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Perplexity Evolution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "F06 — Genuine Training vs Validation Loss Curve (Tiny Shakespeare)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = ROOT / "outputs/figures/F06_loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    dest = save_evidence(
        out_path,
        sources=[
            "experiments/visualize_loss_curves.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/train.py",
        ],
        details={
            "command": "python -m experiments.visualize_loss_curves",
            "kind": "genuine trained-model loss and perplexity curves",
            "total_steps": steps[-1],
            "best_val_loss": best_val,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "history_file": "outputs/training_history.json",
        },
    )
    print(f"Saved F06 evidence to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Generate F06 loss curves")
    parser.add_argument("--history", type=Path, default=None, help="Path to training history JSON")
    args = parser.parse_args()
    generate_loss_figure(args.history)


if __name__ == "__main__":
    main()
