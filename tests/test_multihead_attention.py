"""Scientific checks of projections, head order, causality and trainability."""

from unittest.mock import patch

import pytest
import torch

from experiments.multihead_shape_trace import scalar_multihead_reference, small_example
from zaid_lm.attention import scaled_dot_product_attention
from zaid_lm.config import load_config
from zaid_lm.multihead_attention import MultiHeadSelfAttention


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_baseline_and_exact_engine_reuse():
    config = load_config("configs/baseline.json")
    module = MultiHeadSelfAttention(config.d_model, config.n_heads)
    x = torch.randn(2, 8, 128)
    with patch(
        "zaid_lm.multihead_attention.scaled_dot_product_attention",
        wraps=scaled_dot_product_attention,
    ) as engine:
        output, weights = module(x, return_attention=True)
    assert output.shape == (2, 8, 128)
    assert weights.shape == (2, 4, 8, 8)
    engine.assert_called_once()
    assert all(t.shape == (2, 4, 8, 32) for t in engine.call_args.args)
    assert engine.call_args.kwargs == {"causal": True, "mask": None}
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 8))
    assert module(x).shape == output.shape


@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_configurable_heads_and_scalar_reference(heads):
    module = MultiHeadSelfAttention(8, heads, dtype=torch.float64).eval()
    x = torch.arange(48, dtype=torch.float64).reshape(2, 3, 8) / 17 - 1
    output, weights = module(x, return_attention=True)
    expected, expected_weights = scalar_multihead_reference(x, module, causal=True)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_fixed_asymmetric_reference(bias, causal):
    module, x = small_example(bias=bias)
    output, weights = module(x, causal=causal, return_attention=True)
    expected, expected_weights = scalar_multihead_reference(x, module, causal=causal)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)
    # An incorrect direct reshape (B,H,T,Dh) must be distinguishable on this input.
    projected = module.q_proj(x)
    assert not torch.equal(module._split_heads(projected), projected.reshape(1, 2, 3, 2))


def test_projection_independence_and_output_projection():
    module, x = small_example()
    layers = (module.q_proj, module.k_proj, module.v_proj, module.out_proj)
    assert len({layer.weight.data_ptr() for layer in layers}) == 4
    before = module.k_proj(x).clone()
    with torch.no_grad():
        module.q_proj.weight.add_(0.1)
    torch.testing.assert_close(module.k_proj(x), before, rtol=0, atol=0)
    head_out, _ = scaled_dot_product_attention(
        *(module._split_heads(layer(x)) for layer in layers[:3]), causal=True
    )
    expected = module.out_proj(module._merge_heads(head_out))
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    with torch.no_grad():
        module.out_proj.weight.zero_()
        module.out_proj.bias.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.float64))
    torch.testing.assert_close(module(x), module.out_proj.bias.expand_as(x), rtol=0, atol=0)


def test_split_merge_preserves_token_and_feature_order():
    module = MultiHeadSelfAttention(6, 3)
    x = torch.arange(2 * 4 * 6).reshape(2, 4, 6).float()
    heads = module._split_heads(x)
    for b in range(2):
        for h in range(3):
            for t in range(4):
                torch.testing.assert_close(
                    heads[b, h, t], x[b, t, 2 * h : 2 * h + 2], rtol=0, atol=0
                )
    torch.testing.assert_close(module._merge_heads(heads), x, rtol=0, atol=0)


def test_unmasked_token_permutation_equivariance():
    module, x = small_example()
    order = torch.tensor([2, 0, 1])
    y, weights = module(x, causal=False, return_attention=True)
    permuted, permuted_weights = module(x[:, order], causal=False, return_attention=True)
    torch.testing.assert_close(permuted, y[:, order], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        permuted_weights, weights[:, :, order][:, :, :, order], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_final_output_future_invariance_and_forbidden_weights(dtype):
    module = MultiHeadSelfAttention(8, 2, dtype=dtype).eval()
    x = torch.randn(2, 6, 8, dtype=dtype)
    y, weights = module(x, return_attention=True)
    for prefix in range(1, 6):
        altered = x.clone()
        altered[:, prefix:] = torch.randn_like(altered[:, prefix:]) * 7
        torch.testing.assert_close(module(altered)[:, :prefix], y[:, :prefix], rtol=0, atol=0)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


@pytest.mark.parametrize("bias,expected", [(True, 66048), (False, 65536)])
def test_parameter_count(bias, expected):
    module = MultiHeadSelfAttention(128, 4, bias=bias)
    assert sum(p.numel() for p in module.parameters()) == expected
    assert all(p.requires_grad for p in module.parameters())


def test_input_and_all_projection_gradients_and_gradcheck():
    module, x = small_example()
    x.requires_grad_(True)
    module(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, name
    assert torch.autograd.gradcheck(
        lambda values: module(values),
        (x.detach().requires_grad_(),),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_batch_independence_singleton_and_noncontiguous_input():
    module = MultiHeadSelfAttention(4, 2, dtype=torch.float64)
    x = torch.randn(2, 4, 3, dtype=torch.float64).transpose(1, 2)
    assert not x.is_contiguous()
    y = module(x)
    for b in range(2):
        torch.testing.assert_close(y[b : b + 1], module(x[b : b + 1]), rtol=1e-12, atol=1e-12)
    singleton = x[:, :1]
    single_y, weights = module(singleton, return_attention=True)
    torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)
    torch.testing.assert_close(single_y, module.out_proj(module.v_proj(singleton)), rtol=0, atol=0)


@pytest.mark.parametrize(
    "d_model,heads",
    [(0, 2), (-1, 1), (True, 1), (4.0, 2), (4, 0), (4, -2), (4, True), (4, 1.5), (5, 2)],
)
def test_invalid_dimensions(d_model, heads):
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model, heads)


@pytest.mark.parametrize("shape", [(3, 4), (1, 2, 3, 4), (0, 2, 4), (2, 0, 4), (1, 2, 5)])
def test_invalid_input_shapes(shape):
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(4, 2)(torch.zeros(shape))


@pytest.mark.parametrize("bad", [None, [[1, 2]]])
def test_non_tensor_input(bad):
    with pytest.raises(TypeError):
        MultiHeadSelfAttention(4, 2)(bad)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_invalid_dtype(dtype):
    with pytest.raises(TypeError):
        MultiHeadSelfAttention(4, 2)(torch.ones(1, 2, 4, dtype=dtype))


def test_dtype_and_device_mismatch_without_gpu_work():
    module = MultiHeadSelfAttention(4, 2)
    with pytest.raises(ValueError, match="dtype and device"):
        module(torch.zeros(1, 2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype and device"):
        module(torch.empty(1, 2, 4, device="meta"))
    module = module.double()
    assert module(torch.zeros(1, 2, 4, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input(value):
    with pytest.raises(ValueError, match="finite"):
        MultiHeadSelfAttention(4, 2)(torch.full((1, 2, 4), value))


def test_masks_broadcast_combine_and_invalid_masks():
    module = MultiHeadSelfAttention(4, 2)
    x = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
    mask[0, :, :, 1] = True
    _, weights = module(x, causal=False, mask=mask, return_attention=True)
    assert torch.count_nonzero(weights[0, :, :, 1]) == 0
    additive = torch.zeros_like(mask, dtype=x.dtype).masked_fill(mask, -torch.inf)
    torch.testing.assert_close(
        module(x, causal=False, mask=additive), module(x, causal=False, mask=mask), rtol=0, atol=0
    )
    # Preserve each query's diagonal when combining masks with causal restrictions.
    causal_mask = torch.zeros(3, 3, dtype=torch.bool)
    causal_mask[2, 0] = True
    _, weights = module(x, mask=causal_mask, return_attention=True)
    assert torch.count_nonzero(torch.triu(weights, 1)) == 0
    assert torch.count_nonzero(weights[:, :, 2, 0]) == 0
    for bad in (
        torch.zeros(5, 5, dtype=torch.bool),
        torch.ones(3, 3, dtype=torch.bool),
        torch.ones(3, 3),
        torch.zeros(3, 3, dtype=torch.float64),
    ):
        with pytest.raises((ValueError, TypeError)):
            module(x, mask=bad)


def test_dropout_changes_outputs_but_not_normalized_weights():
    module = MultiHeadSelfAttention(16, 4, dropout=0.5)
    x = torch.randn(2, 6, 16)
    module.eval()
    expected, weights = module(x, return_attention=True)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    module.train()
    y1, w1 = module(x, return_attention=True)
    y2, w2 = module(x, return_attention=True)
    assert not torch.equal(y1, y2)
    assert (y1 == 0).any()
    torch.testing.assert_close(w1, weights, rtol=0, atol=0)
    torch.testing.assert_close(w2, weights, rtol=0, atol=0)
    assert torch.all((y1 == 0) | torch.isclose(y1, expected * 2))


@pytest.mark.parametrize("dropout", [-0.1, 1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_dropout(dropout):
    with pytest.raises((ValueError, TypeError)):
        MultiHeadSelfAttention(4, 2, dropout=dropout)


def test_invalid_flags_and_constructor_dtype():
    with pytest.raises(TypeError):
        MultiHeadSelfAttention(4, 2, bias=1)
    with pytest.raises(TypeError):
        MultiHeadSelfAttention(4, 2, dtype=torch.float16)
    module = MultiHeadSelfAttention(4, 2)
    with pytest.raises(TypeError):
        module(torch.zeros(1, 2, 4), return_attention=1)
    with pytest.raises((TypeError, ValueError)):
        module(torch.zeros(1, 2, 4), causal=1)


def test_nonfinite_output_projection_rejected():
    module = MultiHeadSelfAttention(4, 2)
    with torch.no_grad():
        module.out_proj.weight.fill_(float("inf"))
    with pytest.raises(ValueError, match="output projection"):
        module(torch.ones(1, 2, 4))
