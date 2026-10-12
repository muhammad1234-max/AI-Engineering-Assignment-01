"""Phase 7 (Milestone M6): Controlled ablations on depth, heads, and context."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.config import load_config
from zaid_lm.dataset import prepare_dataset
from zaid_lm.gpt import GPT
from zaid_lm.train import train

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 3 controlled ablations varying exactly one hyperparameter from baseline
ABLATION_CONFIGS = [
    {
        "name": "Baseline (4L, 4H, T128)",
        "overrides": {"n_layers": 4, "n_heads": 4, "context_length": 128},
        "factor": "Baseline",
    },
    {
        "name": "Depth: 2 Layers",
        "overrides": {"n_layers": 2, "n_heads": 4, "context_length": 128},
        "factor": "Depth (Layers)",
    },
    {
        "name": "Heads: 2 Heads",
        "overrides": {"n_layers": 4, "n_heads": 2, "context_length": 128},
        "factor": "Attention Heads",
    },
    {
        "name": "Context: 64 Tokens",
        "overrides": {"n_layers": 4, "n_heads": 4, "context_length": 64},
        "factor": "Context Window",
    },
]


def run_ablations(
    *,
    steps: int = 500,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> Path:
    """Run all controlled ablation experiments and generate F09 comparison figure."""
    if output_dir is None:
        output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(ROOT / "configs/baseline.json")
    splits = prepare_dataset(ROOT / "data")

    results = []
    print("=" * 70)
    print("Phase 7 — Controlled Ablation Study (Depth, Attention Heads, Context)")
    print("=" * 70)

    for exp in ABLATION_CONFIGS:
        name = exp["name"]
        overrides = {
            **exp["overrides"],
            "vocab_size": splits.tokenizer.vocab_size,
            "train_steps": steps,
            "device": device,
        }
        cfg = replace(base_config, **overrides)
        torch.manual_seed(cfg.seed)
        model = GPT(cfg)
        param_count = model.count_parameters()

        print(
            f"\nRunning Variant: {name} | Params: {param_count:,} | Device: {device} | Steps: {steps}..."
        )
        t0 = time.time()

        res = train(
            model=model,
            train_data=splits.train_tokens,
            val_data=splits.val_tokens,
            config=cfg,
            tokenizer=splits.tokenizer,
            eval_interval=steps // 2,
            checkpoint_dir=output_dir / "checkpoints_ablation",
        )
        elapsed = time.time() - t0

        results.append(
            {
                "name": name,
                "factor": exp["factor"],
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "context_length": cfg.context_length,
                "parameters": param_count,
                "best_val_loss": round(res["best_val_loss"], 4),
                "final_train_loss": round(res["history"][-1]["train_loss"], 4),
                "elapsed_seconds": round(elapsed, 1),
            }
        )

    # Save results JSON
    results_file = output_dir / "ablation_results.json"
    results_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Generate F09 comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)

    names = [r["name"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]
    colors = ["#2b5c8f", "#4682b4", "#5c9ebb", "#82b1cf"]

    # 1. Validation Loss Comparison
    bars1 = ax1.bar(names, val_losses, color=colors, edgecolor="black", width=0.55)
    ax1.set_ylabel("Best Validation Loss (nats)", fontsize=11, fontweight="bold")
    ax1.set_title("Validation Loss by Architectural Variant", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)

    for bar, val in zip(bars1, val_losses, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # 2. Parameter vs Efficiency Table
    ax2.axis("off")
    table_data = [
        ["Ablation Variant", "Layers", "Heads", "Context", "Params (k)", "Best Val Loss"],
    ]
    for r in results:
        table_data.append(
            [
                r["name"],
                str(r["n_layers"]),
                str(r["n_heads"]),
                str(r["context_length"]),
                f"{r['parameters'] / 1000:.1f}k",
                f"{r['best_val_loss']:.4f}",
            ]
        )

    table = ax2.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.12, 0.12, 0.14, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#e8f4f8")  # Baseline highlight
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f9f9f9")

    fig.suptitle(
        "F09 — Controlled Architectural Ablation Study (Measured Empirical Results)\n"
        "Single-Factor Perturbations Against Baseline on Tiny Shakespeare",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = ROOT / "outputs/figures/F09_ablations.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    dest = save_evidence(
        out_path,
        sources=[
            "experiments/run_ablations.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/train.py",
            "src/zaid_lm/gpt.py",
        ],
        details={
            "command": "python -m experiments.run_ablations",
            "kind": "controlled ablation study varying depth, heads, and context",
            "variants_tested": len(results),
            "results": results,
        },
    )
    print(f"\nSaved F09 evidence to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Run Phase 7 controlled ablations")
    parser.add_argument("--steps", type=int, default=500, help="Train steps per variant")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    args = parser.parse_args()
    run_ablations(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
