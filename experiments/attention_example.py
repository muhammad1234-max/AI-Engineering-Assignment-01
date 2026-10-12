"""Independent scalar verification of the worked solution; reproducible F01 summary."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from zaid_lm.attention import scaled_dot_product_attention

ROOT = Path(__file__).resolve().parents[1]
Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V = [[1.0, 2.0], [3.0, 0.0], [0.0, 4.0]]


def scalar_reference(q, k, v, *, causal=False):
    """Small independent Python-loop oracle for valid 2D lists, no production helpers.

    Used only for educational checks. Stable row exponentials also permit finite
    large logits. Validation of general tensor inputs belongs to the public engine.
    """
    dot = [[sum(x * y for x, y in zip(qi, kj, strict=True)) for kj in k] for qi in q]
    scores = [[x / math.sqrt(len(q[0])) for x in row] for row in dot]
    masked = [
        [-math.inf if causal and j > i else x for j, x in enumerate(row)]
        for i, row in enumerate(scores)
    ]
    weights = []
    for row in masked:
        maximum = max(row)
        numerators = [math.exp(x - maximum) for x in row]
        denominator = sum(numerators)
        weights.append([x / denominator for x in numerators])
    output = [
        [sum(row[j] * v[j][c] for j in range(len(v))) for c in range(len(v[0]))] for row in weights
    ]
    return {
        "dot_products": dot,
        "scores": scores,
        "masked_scores": masked,
        "weights": weights,
        "output": output,
    }


def manual_reference():
    """Read the literal nine-decimal appendix; null is explicitly the -inf sentinel."""
    text = (ROOT / "docs/manual_attention.md").read_text(encoding="utf-8")
    appendix = text.split("<!-- attention-reference -->", 1)[1]
    block = appendix.split("```json", 1)[1].split("```", 1)[0]
    return {
        name: [[-math.inf if x is None else x for x in row] for row in matrix]
        for name, matrix in json.loads(block).items()
    }


def verify_example():
    """Fail on any manual/scalar/engine mismatch; return unrounded verified matrices."""
    plain = scalar_reference(Q, K, V)
    causal = scalar_reference(Q, K, V, causal=True)
    reference = {name: plain[name] for name in ("dot_products", "scores", "weights", "output")}
    reference.update(
        masked_scores=causal["masked_scores"],
        masked_weights=causal["weights"],
        masked_output=causal["output"],
    )
    manual = manual_reference()
    assert manual.keys() == reference.keys(), "manual appendix must contain all seven matrices"
    q, k, v = [torch.tensor(x, dtype=torch.float64) for x in (Q, K, V)]
    output, weights = scaled_dot_product_attention(q, k, v)
    causal_output, causal_weights = scaled_dot_product_attention(q, k, v, causal=True)
    dot = q @ k.T
    scores = dot / math.sqrt(q.shape[-1])
    # This verification mask is independently built from index comparisons.
    forbidden = torch.tensor([[j > i for j in range(3)] for i in range(3)])
    engine = {
        "dot_products": dot,
        "scores": scores,
        "weights": weights,
        "output": output,
        "masked_scores": scores.masked_fill(forbidden, -math.inf),
        "masked_weights": causal_weights,
        "masked_output": causal_output,
    }
    errors = {}
    for name, matrix in reference.items():
        expected = torch.tensor(matrix, dtype=torch.float64)
        transcribed = torch.tensor(manual[name], dtype=torch.float64)
        torch.testing.assert_close(transcribed, expected, rtol=0, atol=5.1e-10)
        torch.testing.assert_close(engine[name], expected, rtol=1e-12, atol=1e-12)
        finite = torch.isfinite(expected)
        errors[name] = {
            "manual_max_absolute_error": (transcribed[finite] - expected[finite])
            .abs()
            .max()
            .item(),
            "engine_max_absolute_error": (engine[name][finite] - expected[finite])
            .abs()
            .max()
            .item(),
        }
    for name in ("weights", "masked_weights"):
        assert all(abs(sum(row) - 1) <= 1e-12 for row in reference[name])
    assert all(reference["masked_weights"][i][j] == 0 for i in range(3) for j in range(i + 1, 3))
    return reference, errors


def save_evidence(path: Path, *, sources: list[str], details: dict):
    """Copy only a small validated PNG and a content-hash provenance receipt."""
    content = path.read_bytes()
    if not content.startswith(b"\x89PNG\r\n\x1a\n") or len(content) > 500_000:
        raise ValueError("evidence must be a PNG no larger than 500 KB")
    destination = ROOT / "report/evidence" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    receipt = {
        "artifact": destination.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "details": details,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def generate_f01(reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = [
        ("Q (3×2)", Q),
        ("K (3×2)", K),
        ("V (3×2)", V),
        ("QKᵀ (3×3)", reference["dot_products"]),
        ("S = QKᵀ / √2 (3×3)", reference["scores"]),
        ("A = row softmax(S) (3×3)", reference["weights"]),
        ("O = AV (3×2)", reference["output"]),
        ("Causal A (3×3)", reference["masked_weights"]),
        ("Causal O (3×2)", reference["masked_output"]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    for ax, (title, matrix) in zip(axes.flat, matrices, strict=True):
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=12)
        table = ax.table(
            cellText=[[f"{x:.6f}" for x in row] for row in matrix],
            rowLabels=["t1", "t2", "t3"],
            colLabels=[f"key t{i + 1}" for i in range(3)]
            if len(matrix[0]) == 3
            else ["feature 1", "feature 2"],
            loc="center",
            cellLoc="center",
            bbox=[0.08, 0.2, 0.90, 0.60],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    fig.suptitle(
        "F01 — Verified three-token worked calculation\ndₖ = dᵥ = 2; six decimals; compute with unrounded values",
        fontsize=15,
    )
    fig.supxlabel(
        "Typed derivation: docs/manual_attention.md · Scalar-loop and float64 engine verified\nProgrammatic summary; Zaid's personal arithmetic review remains required",
        fontsize=10,
    )
    path = ROOT / "outputs/figures/F01_manual_attention.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, metadata={"Software": "Zaid Phase 2 verification script"})
    plt.close(fig)
    save_evidence(
        path,
        sources=[
            "experiments/attention_example.py",
            "src/zaid_lm/attention.py",
            "docs/manual_attention.md",
        ],
        details={
            "kind": "verified worked calculation, not handwriting or trained results",
            "rounding": "six decimals",
            "human_review": "pending",
        },
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="store_true", help="regenerate and copy verified F01")
    args = parser.parse_args()
    reference, errors = verify_example()
    serializable = {
        name: [[None if not math.isfinite(x) else x for x in row] for row in matrix]
        for name, matrix in reference.items()
    }
    receipt = {
        "status": "PASS",
        "dtype": "float64",
        "device": "cpu",
        "manual_atol": 5.1e-10,
        "engine_rtol": 1e-12,
        "engine_atol": 1e-12,
        "q": Q,
        "k": K,
        "v": V,
        "matrices": serializable,
        "errors": errors,
        "human_arithmetic_review": "pending",
    }
    if args.figure:
        receipt["figure"] = generate_f01(reference).relative_to(ROOT).as_posix()
        destination = ROOT / "report/evidence/attention_verification.json"
        destination.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
