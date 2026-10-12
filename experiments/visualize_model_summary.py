"""Generate F05: Model summary and parameter count breakdown for baseline GPT."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.config import load_config
from zaid_lm.gpt import GPT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_summary_figure(config_path: Path | None = None) -> Path:
    """Generate F05 model summary figure and provenance receipt."""
    if config_path is None:
        config_path = ROOT / "configs/baseline.json"

    base_config = load_config(config_path)
    # Character tokenizer on Tiny Shakespeare has ~65 unique characters
    config = replace(base_config, vocab_size=65)

    torch.manual_seed(42)
    model = GPT(config)
    model.eval()

    total_params = model.count_parameters(trainable_only=True)
    breakdown = model.parameter_breakdown()

    # Aggregate categories
    token_emb = breakdown["token_embedding"]
    assert breakdown["position_encoding"] == 0  # 0 for sinusoidal
    ln_params = breakdown["final_norm"]
    mha_params = 0
    ffn_params = 0

    for i in range(config.n_layers):
        block = breakdown[f"block_{i}"]
        ln_params += block["ln1"] + block["ln2"]
        mha_params += block["mha"]
        ffn_params += block["ffn"]

    lm_head = breakdown["lm_head"]

    categories = [
        ("Token Embedding", token_emb, "#4C72B0"),
        ("Multi-Head Attn", mha_params, "#55A868"),
        ("Feed-Forward Net", ffn_params, "#C44E52"),
        ("Layer Normalization", ln_params, "#8172B2"),
        ("LM Head", lm_head, "#CCB974"),
    ]

    fig = plt.figure(figsize=(12, 7), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.3)

    # 1. Bar chart of parameter distribution
    ax1 = fig.add_subplot(gs[0, :])
    cat_names = [c[0] for c in categories]
    cat_counts = [c[1] for c in categories]
    cat_colors = [c[2] for c in categories]

    bars = ax1.bar(
        cat_names, cat_counts, color=cat_colors, edgecolor="black", linewidth=0.8, width=0.55
    )
    ax1.set_ylabel("Parameter Count", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Baseline GPT Architecture Parameter Breakdown (Total: {total_params:,} parameters)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, cat_counts, strict=True):
        pct = (count / total_params) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (total_params * 0.015),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(cat_counts) * 1.25)

    # 2. Layer Table
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")

    table_data = [
        ["Layer / Component", "Parameters", "% of Total"],
        ["Token Embedding (V×d)", f"{token_emb:,}", f"{token_emb / total_params * 100:.1f}%"],
        ["Positional Encoding", "0 (sinusoidal)", "0.0%"],
        [
            f"Attention ({config.n_layers} blocks)",
            f"{mha_params:,}",
            f"{mha_params / total_params * 100:.1f}%",
        ],
        [
            f"FFN ({config.n_layers} blocks)",
            f"{ffn_params:,}",
            f"{ffn_params / total_params * 100:.1f}%",
        ],
        [
            f"LayerNorms (2×{config.n_layers}+1)",
            f"{ln_params:,}",
            f"{ln_params / total_params * 100:.1f}%",
        ],
        ["LM Head (d×V)", f"{lm_head:,}", f"{lm_head / total_params * 100:.1f}%"],
        ["TOTAL TRAINABLE", f"{total_params:,}", "100.0%"],
    ]

    table = ax2.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.28, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif row == len(table_data) - 1:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f9f9f9")

    # 3. Model Hyperparameters & Specs
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    specs_text = (
        r"$\bf{Model\ Hyperparameters:}$" + "\n\n"
        f"• Context Length ($T_{{max}}$): {config.context_length}\n"
        f"• Model Dimension ($d_{{model}}$): {config.d_model}\n"
        f"• Attention Heads ($n_{{heads}}$): {config.n_heads} (head dim: {config.d_model // config.n_heads})\n"
        f"• Decoder Layers ($n_{{layers}}$): {config.n_layers}\n"
        f"• FFN Hidden Dimension ($d_{{ffn}}$): {config.ffn_dim} (4 × $d_{{model}}$)\n"
        f"• Vocabulary Size ($V$): {config.vocab_size} (TinyShakespeare char)\n"
        f"• Dropout Rate: {config.dropout}\n"
        f"• Activation: {config.activation.upper()}\n"
        f"• Position Encoding: {config.position_encoding.title()}\n\n"
        f"• Model Size (fp32): ~{total_params * 4 / 1024 / 1024:.2f} MB\n"
        f"• Architecture: Pre-LN Causal Decoder-Only Transformer"
    )
    ax3.text(
        0.05,
        0.95,
        specs_text,
        transform=ax3.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f8f9fa", "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        "F05 — Baseline GPT Model Summary and Parameter Count",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    path = ROOT / "outputs/figures/F05_model_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=140, bbox_inches="tight", metadata={"Software": "Zaid Phase 4 GPT model summary"}
    )
    plt.close(fig)

    destination = save_evidence(
        path,
        sources=[
            "experiments/visualize_model_summary.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/gpt.py",
            "src/zaid_lm/transformer_block.py",
            "src/zaid_lm/config.py",
        ],
        details={
            "command": "python -m experiments.visualize_model_summary",
            "kind": "baseline model parameter breakdown and summary",
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "ffn_dim": config.ffn_dim,
            "total_parameters": total_params,
            "breakdown": breakdown,
        },
    )
    print(f"Saved evidence F05 to {destination}")
    print(f"Total model parameters: {total_params:,}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Generate F05 model summary figure")
    parser.add_argument("--config", type=Path, default=None, help="Path to config file")
    args = parser.parse_args()
    generate_summary_figure(args.config)


if __name__ == "__main__":
    main()
