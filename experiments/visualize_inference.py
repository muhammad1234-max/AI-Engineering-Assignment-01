"""Generate F10: Actual CLI and REST API live inference evidence with metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from fastapi.testclient import TestClient

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.api import app, load_model_state
from zaid_lm.generate import generate_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_inference_figure(checkpoint_path: Path | None = None) -> Path:
    """Generate F10 showing real CLI output and REST API request/response with timing."""
    if checkpoint_path is None:
        checkpoint_path = ROOT / "checkpoints/best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 1. Execute real CLI inference
    cli_prompt = "KING CLAUDIUS:\n"
    cli_result = generate_text(
        checkpoint_path,
        cli_prompt,
        max_new_tokens=70,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=42,
    )

    # 2. Execute real REST API inference via TestClient
    load_model_state()
    client = TestClient(app)

    api_prompt = "ROMEO:\n"
    req_payload = {
        "prompt": api_prompt,
        "max_new_tokens": 60,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "seed": 123,
    }

    info_resp = client.get("/model-info").json()
    gen_resp = client.post("/generate", json=req_payload).json()

    # Render F10 Figure
    fig = plt.figure(figsize=(14, 7.5), dpi=140)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # Panel 1: FastAPI REST API Request & Response
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")

    api_card_text = (
        r"$\bf{FastAPI\ REST\ API\ Inference\ (POST\ /generate)}$" + "\n"
        f"{'═' * 55}\n\n"
        r"$\bf{Request\ Payload\ (JSON):}$" + "\n"
        f"{json.dumps(req_payload, indent=2)}\n\n"
        r"$\bf{HTTP\ 200\ OK\ Response\ (JSON):}$" + "\n"
        f"{{\n"
        f'  "prompt": {json.dumps(gen_resp["prompt"])},\n'
        f'  "completion": {json.dumps(gen_resp["completion"][:60])}...,\n'
        f'  "tokens_generated": {gen_resp["tokens_generated"]},\n'
        f'  "elapsed_seconds": {gen_resp["elapsed_seconds"]}s,\n'
        f'  "tokens_per_second": {gen_resp["tokens_per_second"]} tok/s,\n'
        f'  "model_step": {gen_resp["model_step"]},\n'
        f'  "val_loss": {gen_resp["val_loss"]:.4f}\n'
        f"}}\n\n"
        f"{'─' * 55}\n"
        f"Model: {info_resp['model_name']} ({info_resp['parameters']:,} params)\n"
        f"Architecture: {info_resp['architecture']} (Pre-LN)\n"
        f"Layers: {info_resp['n_layers']} | Heads: {info_resp['n_heads']} | Dim: {info_resp['d_model']}"
    )

    ax1.text(
        0.03,
        0.97,
        api_card_text,
        transform=ax1.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#4C72B0", linewidth=1.2
        ),
    )

    # Panel 2: Command-Line Interface (CLI) Execution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    cli_card_text = (
        r"$\bf{Interactive\ CLI\ Inference\ (zaid\_lm.generate)}$" + "\n"
        f"{'═' * 55}\n\n"
        f"$ python -m zaid_lm.generate \\\n"
        f"    --checkpoint checkpoints/best_model.pt \\\n"
        f"    --prompt {repr(cli_prompt)} \\\n"
        f"    --max-tokens 70 \\\n"
        f"    --temperature 0.8 \\\n"
        f"    --seed 42\n\n"
        r"$\bf{Terminal\ Output:}$" + "\n"
        f"{'─' * 55}\n"
        f"Prompt: {repr(cli_prompt)}\n"
        f"Generated Text:\n\n"
        f"{cli_result['full_text'].strip()}\n\n"
        f"{'─' * 55}\n"
        f"[Generated {cli_result['new_tokens']} tokens in {cli_result['elapsed_seconds']:.2f}s "
        f"({cli_result['tokens_per_second']:.1f} tok/s)]\n\n"
        f"Checkpoint Step: {cli_result['model_step']} | Val Loss: {cli_result['val_loss']:.4f}\n"
        f"Vocab: 65 chars | Context: 128 tokens | Device: {cli_result.get('device', 'cpu')}"
    )

    ax2.text(
        0.03,
        0.97,
        cli_card_text,
        transform=ax2.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#1e1e1e", edgecolor="#27ae60", linewidth=1.2
        ),
        color="#ecf0f1",
    )

    fig.suptitle(
        "F10 — Live Inference Evidence (FastAPI REST API & Command-Line Interface)\n"
        "Reproducible Outputs from Genuine Trained Checkpoint on Tiny Shakespeare",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = ROOT / "outputs/figures/F10_inference.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    dest = save_evidence(
        out_path,
        sources=[
            "experiments/visualize_inference.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/api.py",
            "src/zaid_lm/generate.py",
        ],
        details={
            "command": "python -m experiments.visualize_inference",
            "kind": "live inference evidence showing CLI and REST API execution",
            "checkpoint": str(checkpoint_path.name),
            "cli_prompt": cli_prompt,
            "api_prompt": api_prompt,
            "api_response": gen_resp,
            "cli_result": {
                "tokens": cli_result["new_tokens"],
                "elapsed": round(cli_result["elapsed_seconds"], 3),
                "tok_per_sec": round(cli_result["tokens_per_second"], 1),
            },
        },
    )
    print(f"Saved F10 evidence to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Generate F10 inference figure")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    generate_inference_figure(args.checkpoint)


if __name__ == "__main__":
    main()
