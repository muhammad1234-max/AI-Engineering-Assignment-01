"""Executed F03 shape trace and independent scalar-loop projection verification."""

import json
import math

import torch

from experiments.attention_example import scalar_reference
from experiments.phase3_evidence import ROOT, save_evidence
from zaid_lm.attention import scaled_dot_product_attention
from zaid_lm.config import ProjectConfig
from zaid_lm.multihead_attention import MultiHeadSelfAttention
from zaid_lm.reproducibility import seed_everything


def scalar_multihead_reference(x, module, *, causal):
    """Python loops for projections, feature slicing, attention and output projection.

    Uses read-only copies of weights, never the module forward or tensor matmul.
    Only valid small educational examples without custom masks/dropout are intended.
    """

    def linear(rows, layer):
        weight = layer.weight.detach().tolist()
        bias = layer.bias.detach().tolist() if layer.bias is not None else [0.0] * len(weight)
        return [
            [sum(a * b for a, b in zip(row, w, strict=True)) + c for w, c in zip(weight, bias)]
            for row in rows
        ]

    results, all_weights = [], []
    for batch in x.detach().tolist():
        q, k, v = [linear(batch, layer) for layer in (module.q_proj, module.k_proj, module.v_proj)]
        head_results, head_weights = [], []
        for h in range(module.n_heads):
            start, stop = h * module.d_head, (h + 1) * module.d_head
            head = scalar_reference(
                [row[start:stop] for row in q],
                [row[start:stop] for row in k],
                [row[start:stop] for row in v],
                causal=causal,
            )
            head_results.append(head["output"])
            head_weights.append(head["weights"])
        merged = [
            [feature for head in head_results for feature in head[t]] for t in range(len(batch))
        ]
        results.append(linear(merged, module.out_proj))
        all_weights.append(head_weights)
    return torch.tensor(results, dtype=x.dtype), torch.tensor(all_weights, dtype=x.dtype)


def small_example(*, bias=True):
    module = MultiHeadSelfAttention(4, 2, bias=bias, dtype=torch.float64).eval()
    matrices = [
        [[1, 0.2, -0.1, 0.3], [0.4, 1, 0.1, -0.2], [0, -0.3, 1, 0.2], [0.1, 0, 0.4, 1]],
        [[0.5, -0.2, 0.3, 0], [0.1, 0.8, -0.4, 0.2], [0.2, 0, 0.7, -0.1], [-0.3, 0.1, 0, 0.9]],
        [[0.8, 0.1, 0, -0.2], [0, 0.6, 0.3, 0.1], [-0.1, 0.2, 1.1, 0], [0.3, 0, -0.2, 0.7]],
        [[1, 0.2, -0.3, 0.1], [0.4, 0.7, 0.2, 0], [0, -0.1, 0.9, 0.3], [-0.2, 0.1, 0.4, 1]],
    ]
    with torch.no_grad():
        for index, (layer, matrix) in enumerate(
            zip(
                (module.q_proj, module.k_proj, module.v_proj, module.out_proj),
                matrices,
                strict=True,
            )
        ):
            layer.weight.copy_(torch.tensor(matrix, dtype=torch.float64))
            if layer.bias is not None:
                layer.bias.copy_(
                    torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float64) * (index + 1)
                )
    x = torch.tensor(
        [[[1, -2, 0.5, 3], [-0.5, 1, 2, -1], [2, 0.25, -1, 0.75]]], dtype=torch.float64
    )
    return module, x


def verify_reference():
    module, x = small_example()
    errors = {}
    for causal in (False, True):
        output, weights = module(x, causal=causal, return_attention=True)
        expected, expected_weights = scalar_multihead_reference(x, module, causal=causal)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)
        errors[str(causal)] = {
            "output_max_absolute_error": (output - expected).abs().max().item(),
            "weights_max_absolute_error": (weights - expected_weights).abs().max().item(),
            "output": output.detach().tolist(),
        }
    return errors


def runtime_trace():
    config = ProjectConfig()
    seed_everything(config.seed, device="cpu")
    module = MultiHeadSelfAttention(config.d_model, config.n_heads).eval()
    x = torch.randn(2, 8, config.d_model)
    projections, merged = {}, []
    handles = []
    for name in ("q", "k", "v"):
        handles.append(
            getattr(module, f"{name}_proj").register_forward_hook(
                lambda layer, args, output, name=name: projections.update({name: output.detach()})
            )
        )
    handles.append(
        module.out_proj.register_forward_pre_hook(
            lambda layer, args: merged.append(args[0].detach())
        )
    )
    try:
        output, weights = module(x, return_attention=True)
    finally:
        for handle in handles:
            handle.remove()
    tensors = {"Input X": x}
    permuted = []
    for name in ("q", "k", "v"):
        projected = projections[name]
        tensors[f"{name.upper()} projection"] = projected
        split = projected.reshape(2, 8, config.n_heads, module.d_head)
        tensors[f"{name.upper()} split"] = split
        tensor = module._split_heads(projected)
        tensors[f"{name.upper()} permuted"] = tensor
        permuted.append(tensor)
    q, k, v = permuted
    tensors["Per-head scores (inspection computation)"] = (
        q @ k.transpose(-2, -1) / math.sqrt(module.d_head)
    )
    # Inspection re-executes the unchanged engine on captured real projections.
    heads, inspected_weights = scaled_dot_product_attention(q, k, v, causal=True)
    torch.testing.assert_close(inspected_weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(module._merge_heads(heads), merged[0], rtol=0, atol=0)
    torch.testing.assert_close(module.out_proj(merged[0]), output, rtol=0, atol=0)
    tensors.update(
        {
            "Attention weights": weights,
            "Head outputs": heads,
            "Merged outputs (captured projection input)": merged[0],
            "Final output projection": output,
        }
    )
    lines = [
        "F03: EXECUTED MULTI-HEAD SELF-ATTENTION TRACE",
        "Command: python -m experiments.multihead_shape_trace",
        f"seed={config.seed}; CPU; float32; eval; B=2 T=8 d_model=128 n_heads=4 d_head=32",
    ]
    for name, tensor in tensors.items():
        if "split" in name:
            expected = (2, 8, 4, 32)
        elif "permuted" in name or name == "Head outputs":
            expected = (2, 4, 8, 32)
        elif "scores" in name or name == "Attention weights":
            expected = (2, 4, 8, 8)
        else:
            expected = (2, 8, 128)
        assert tuple(tensor.shape) == expected, (name, tensor.shape, expected)
        lines.append(f"{name}: {tuple(tensor.shape)} PASS")
    count = sum(p.numel() for p in module.parameters())
    assert count == 4 * (config.d_model**2 + config.d_model) == 66048
    lines.append(f"Parameters: {count} = 4 * (128^2 + 128); module only PASS")
    future = x.clone()
    future[:, 4:] += 20
    causal_error = (module(future)[:, :4] - output[:, :4]).abs().max().item()
    assert causal_error == 0
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    lines.append(
        f"Final-output future perturbation max error: {causal_error}; future weights zero PASS"
    )
    loss = output.square().mean()
    loss.backward()
    gradients = {
        name: layer.weight.grad.norm().item()
        for name, layer in (
            ("Q", module.q_proj),
            ("K", module.k_proj),
            ("V", module.v_proj),
            ("O", module.out_proj),
        )
    }
    assert all(math.isfinite(value) and value > 0 for value in gradients.values())
    lines.append("Projection gradient norms: " + json.dumps(gradients, sort_keys=True) + " PASS")
    lines.append("Independent float64 per-head scalar reference (rtol=atol=1e-12):")
    lines.append(json.dumps(verify_reference(), indent=2))
    lines.append(
        "PASS: runtime shapes, head order, projections, parameter count, causal output and gradients verified."
    )
    return "\n".join(lines) + "\n"


def main():
    torch.set_num_threads(1)
    text = runtime_trace()
    path = ROOT / "outputs/logs/F03_multihead_shape_trace.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    save_evidence(
        path,
        sources=[
            "experiments/multihead_shape_trace.py",
            "experiments/phase3_evidence.py",
            "experiments/attention_example.py",
            "src/zaid_lm/multihead_attention.py",
            "src/zaid_lm/attention.py",
            "src/zaid_lm/config.py",
            "src/zaid_lm/reproducibility.py",
        ],
        details={
            "command": "python -m experiments.multihead_shape_trace",
            "kind": "random initialization and fixed small mathematical reference; no model training",
        },
    )
    print(text, end="")


if __name__ == "__main__":
    main()
