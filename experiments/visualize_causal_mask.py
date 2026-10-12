"""Generate F02 from the real bool mask: structural visibility, not trained attention."""

import matplotlib
import torch

from experiments.attention_example import ROOT, save_evidence
from zaid_lm.attention import causal_forbidden_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.80, wspace=0.25)
    for ax, length in zip(axes, (3, 8), strict=True):
        forbidden = causal_forbidden_mask(length)
        independent = torch.tensor([[j > i for j in range(length)] for i in range(length)])
        assert torch.equal(forbidden, independent), "implementation mask differs from index rule"
        ax.imshow(forbidden.numpy(), cmap=ListedColormap(["#cce8d4", "#e8b5b5"]), vmin=0, vmax=1)
        ax.set_title(f"{length} tokens; diagonal allowed")
        ax.set_xlabel("Key token (column)")
        ax.set_ylabel("Query token (row)")
        ax.set_xticks(range(length), [f"t{i + 1}" for i in range(length)])
        ax.set_yticks(range(length), [f"t{i + 1}" for i in range(length)])
        for i in range(length):
            for j in range(length):
                ax.text(
                    j, i, "×" if forbidden[i, j] else "✓", ha="center", va="center", fontsize=14
                )
    fig.suptitle(
        "F02 — Causal visibility from the attention implementation\nNo query attends to a future key (j > i)",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Patch(color="#cce8d4", label="Allowed: bool False / additive 0"),
            Patch(color="#e8b5b5", label="Forbidden: bool True / additive −∞"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.supxlabel(
        "Structural mask only; not trained-model attention (F07 remains future work)",
        fontsize=10,
        y=0.12,
    )
    path = ROOT / "outputs/figures/F02_causal_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
        metadata={"Software": "Zaid Phase 2 verification script"},
    )
    plt.close(fig)
    save_evidence(
        path,
        sources=[
            "experiments/visualize_causal_mask.py",
            "experiments/attention_example.py",
            "src/zaid_lm/attention.py",
        ],
        details={
            "kind": "structural causal mask",
            "lengths": [3, 8],
            "rule": "True blocks key j > query i; diagonal allowed",
        },
    )
    print(f"PASS: {path.relative_to(ROOT)}; copied to report/evidence with SHA256 provenance")


if __name__ == "__main__":
    main()
