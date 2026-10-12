"""Verify every baseline sinusoidal feature against scalar math and generate F04."""

import json
import math

import torch

from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.config import ProjectConfig
from zaid_lm.positional_encoding import SinusoidalPositionalEncoding


def scalar_positions(context_length, d_model):
    return [
        [
            (math.sin if dimension % 2 == 0 else math.cos)(
                position / 10000 ** ((dimension - dimension % 2) / d_model)
            )
            for dimension in range(d_model)
        ]
        for position in range(context_length)
    ]


def verified_positions():
    config = ProjectConfig()
    assert config.position_encoding == "sinusoidal"
    errors = {}
    table = None
    for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
        module = SinusoidalPositionalEncoding(
            config.d_model, config.context_length, dtype=dtype
        ).eval()
        actual = module(torch.zeros(1, config.context_length, config.d_model, dtype=dtype))[0]
        expected = torch.tensor(
            scalar_positions(config.context_length, config.d_model), dtype=torch.float64
        )
        error = (actual.double() - expected).abs().max().item()
        assert error <= tolerance
        assert torch.isfinite(actual).all() and (actual.abs() <= 1).all()
        torch.testing.assert_close(
            actual[0, 0::2], torch.zeros_like(actual[0, 0::2]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[0, 1::2], torch.ones_like(actual[0, 1::2]), rtol=0, atol=0
        )
        assert sum(p.numel() for p in module.parameters()) == 0
        errors[str(dtype)] = {"max_absolute_error": error, "atol": tolerance}
        if dtype == torch.float32:
            table = actual.numpy()
    return table, errors


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, errors = verified_positions()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", width_ratios=[1.3, 1])
    image = axes[0].imshow(table, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set(
        xlabel="Embedding dimension",
        ylabel="Absolute token position",
        title="Implemented position table: 128 positions × 128 features",
    )
    fig.colorbar(image, ax=axes[0], label="Fixed sinusoidal value")
    for dimension in (0, 2, 16, 64):
        axes[1].plot(range(128), table[:, dimension], label=f"sin, dim {dimension}")
    axes[1].set(
        xlabel="Absolute token position",
        ylabel="Encoding value",
        ylim=(-1.1, 1.1),
        title="Higher dimensions change more slowly",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("F04 — Sinusoidal positional encoding from Zaid's implementation", fontsize=14)
    fig.supxlabel(
        "Even: sin(pos / 10000^(2i/128)); odd: cos at the same frequency. Fixed values; no training.",
        fontsize=10,
    )
    path = ROOT / "outputs/figures/F04_positional_encoding.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, metadata={"Software": "Zaid Phase 3 position verification"})
    plt.close(fig)
    destination = save_evidence(
        path,
        sources=[
            "experiments/visualize_positions.py",
            "experiments/phase3_evidence.py",
            "src/zaid_lm/positional_encoding.py",
            "src/zaid_lm/config.py",
        ],
        details={
            "command": "python -m experiments.visualize_positions",
            "kind": "fixed positional encoding, not trained-model evidence",
            "shape": [128, 128],
            "errors": errors,
            "range": [float(table.min()), float(table.max())],
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": destination.relative_to(ROOT).as_posix(),
                "errors": errors,
                "range": [float(table.min()), float(table.max())],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
