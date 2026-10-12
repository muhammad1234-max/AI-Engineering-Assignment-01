"""Independent numerical, causal, gradient and public-contract checks (CPU only)."""

import math

import pytest
import torch

from experiments.attention_example import K, Q, V, scalar_reference, verify_example
from zaid_lm.attention import causal_forbidden_mask, scaled_dot_product_attention


@pytest.fixture(scope="module", autouse=True)
def small_cpu_test_thread_budget():
    """Avoid thread-launch overhead in tiny gradchecks; restore the user's setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(dtype=torch.float64):
    return tuple(torch.tensor(x, dtype=dtype) for x in (Q, K, V))


def random_inputs(shape, *, key_length=None, value_width=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(42)
    leading, length, width = shape[:-2], shape[-2], shape[-1]
    key_length = length if key_length is None else key_length
    return tuple(
        torch.randn(s, generator=generator, dtype=dtype)
        for s in (shape, (*leading, key_length, width), (*leading, key_length, value_width))
    )


def close(actual, expected):
    expected = torch.as_tensor(expected, dtype=actual.dtype)
    tolerance = 1e-6 if actual.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_canonical_all_seven_intermediates_and_manual_transcription():
    reference, errors = verify_example()
    assert len(reference) == 7
    assert max(e["engine_max_absolute_error"] for e in errors.values()) < 1e-12
    assert max(e["manual_max_absolute_error"] for e in errors.values()) <= 5.1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("causal", [False, True])
def test_canonical_engine_matches_scalar_reference(dtype, causal):
    q, k, v = example(dtype)
    output, weights = scaled_dot_product_attention(q, k, v, causal=causal)
    reference = scalar_reference(Q, K, V, causal=causal)
    close(output, reference["output"])
    close(weights, reference["weights"])
    close(weights.sum(-1), torch.ones(3))


@pytest.mark.parametrize(
    "shape,key_length,value_width,causal",
    [
        ((1, 1), 1, 1, True),
        ((5, 7), 5, 4, True),
        ((2, 3, 2), 3, 5, True),
        ((2, 4, 3, 2), 3, 5, True),
        ((3, 2), 7, 4, False),
        ((2, 1, 5), 4, 7, False),
        ((2, 3, 4, 8), 2, 1, False),
    ],
)
def test_general_shapes_normalization_and_batch_head_independence(
    shape, key_length, value_width, causal
):
    q, k, v = random_inputs(shape, key_length=key_length, value_width=value_width)
    output, weights = scaled_dot_product_attention(q, k, v, causal=causal)
    assert output.shape == (*shape[:-2], shape[-2], value_width)
    assert weights.shape == (*shape[:-2], shape[-2], key_length)
    close(weights.sum(-1), torch.ones(weights.shape[:-1]))
    flat_q = q.reshape(-1, shape[-2], shape[-1])
    flat_k = k.reshape(-1, key_length, shape[-1])
    flat_v = v.reshape(-1, key_length, value_width)
    for index, (qi, ki, vi) in enumerate(zip(flat_q, flat_k, flat_v, strict=True)):
        reference = scalar_reference(qi.tolist(), ki.tolist(), vi.tolist(), causal=causal)
        close(output.reshape(-1, shape[-2], value_width)[index], reference["output"])
        close(weights.reshape(-1, shape[-2], key_length)[index], reference["weights"])


def test_scale_is_key_feature_width_not_sequence_length():
    q = torch.tensor([[1.0, 2.0, -1.0, 3.0]], dtype=torch.float64)
    k = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float64
    )
    v = torch.eye(3, dtype=torch.float64)
    output, weights = scaled_dot_product_attention(q, k, v)
    denominator = math.exp(0.5) + math.exp(1) + math.exp(-0.5)
    expected = [
        [math.exp(0.5) / denominator, math.exp(1) / denominator, math.exp(-0.5) / denominator]
    ]
    close(weights, expected)
    close(output, expected)


@pytest.mark.parametrize("length", [1, 2, 3, 8, 128])
def test_causal_mask_structure(length):
    mask = causal_forbidden_mask(length)
    assert mask.dtype == torch.bool and mask.shape == (length, length)
    for i in range(length):
        assert mask[i].tolist() == [j > i for j in range(length)]


def test_causal_zero_weights_and_future_perturbation_invariance():
    q, k, v = random_inputs((2, 4, 5, 3))
    output, weights = scaled_dot_product_attention(q, k, v, causal=True)
    forbidden = causal_forbidden_mask(5).expand_as(weights)
    assert torch.count_nonzero(weights[forbidden]) == 0
    for last_visible in range(4):
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[..., last_visible + 1 :, :] += 100
        changed_v[..., last_visible + 1 :, :] -= 100
        changed_output, _ = scaled_dot_product_attention(q, changed_k, changed_v, causal=True)
        close(changed_output[..., : last_visible + 1, :], output[..., : last_visible + 1, :])


@pytest.mark.parametrize("causal", [False, True])
def test_finite_nonzero_gradients_and_gradcheck(causal):
    inputs = tuple(x.requires_grad_() for x in random_inputs((3, 2), value_width=2))
    output, weights = scaled_dot_product_attention(*inputs, causal=causal)
    loss = output.square().sum() + weights.square().sum()
    loss.backward()
    for x in inputs:
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=causal),
        inputs,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_earlier_output_has_exactly_zero_future_key_value_gradients():
    q, k, v = tuple(x.requires_grad_() for x in example())
    output, _ = scaled_dot_product_attention(q, k, v, causal=True)
    output[1].sum().backward()
    assert torch.equal(v.grad[2], torch.zeros(2, dtype=v.dtype))
    assert torch.equal(k.grad[2], torch.zeros(2, dtype=k.dtype))
    assert v.grad[:2].abs().sum() > 0


def test_singleton_output_equals_value():
    q = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    v = torch.tensor([[7.0, 4.0, -9.0]], dtype=torch.float64)
    output, weights = scaled_dot_product_attention(q, -q, v, causal=True)
    close(output, v)
    close(weights, [[1.0]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["zero", "identical", "negative", "mixed", "large"])
def test_numeric_edge_cases_stable_against_scalar_reference(dtype, kind):
    q, k, v = example(dtype)
    if kind == "zero":
        q.zero_()
        k.zero_()
    elif kind == "identical":
        q.fill_(2)
        k.fill_(3)
    elif kind == "negative":
        q = -q - 1
        k = -k - 2
        v = -v
    elif kind == "mixed":
        q[0, 0] = -2
        k[2, 1] = -3
        v[1, 0] = -5
    elif kind == "large":
        q *= 1e4
        k *= 1e4
    output, weights = scaled_dot_product_attention(q, k, v)
    reference = scalar_reference(q.tolist(), k.tolist(), v.tolist())
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    close(output, reference["output"])
    close(weights, reference["weights"])
    close(weights.sum(-1), torch.ones(3))
    if kind in ("zero", "identical"):
        close(weights, torch.full((3, 3), 1 / 3, dtype=dtype))


@pytest.mark.parametrize("mask_shape", [(3, 3), (1, 3), (2, 1, 3, 3), (1, 4, 1, 3)])
def test_boolean_additive_broadcast_masks_agree(mask_shape):
    q, k, v = random_inputs((2, 4, 3, 2))
    mask = torch.zeros(mask_shape, dtype=torch.bool)
    mask[..., -1] = True
    additive = torch.zeros(mask_shape, dtype=q.dtype).masked_fill(mask, -math.inf)
    output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    output2, weights2 = scaled_dot_product_attention(q, k, v, mask=additive)
    close(output, output2)
    close(weights, weights2)
    assert torch.count_nonzero(weights[..., -1]) == 0
    close(weights.sum(-1), torch.ones(weights.shape[:-1]))


def test_explicit_and_causal_masks_combine_by_union():
    q, k, v = example()
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[1:, 0] = True
    output, weights = scaled_dot_product_attention(q, k, v, causal=True, mask=mask)
    close(weights[0], [1, 0, 0])
    close(weights[1], [0, 1, 0])
    assert weights[2, 0] == 0
    close(output[1], v[1])


@pytest.mark.parametrize(
    "axis,shape,message",
    [
        (0, (3, 4), "feature"),
        (2, (4, 2), "sequence"),
        (0, (0, 2), "positive"),
        (1, (3, 0), "positive"),
        (2, (3, 0), "positive"),
        (0, (2,), "two axes"),
        (0, (1, 3, 2), "leading"),
        (0, (0, 3, 2), "positive"),
    ],
)
def test_invalid_shapes(axis, shape, message):
    inputs = list(example())
    inputs[axis] = torch.empty(shape, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        scaled_dot_product_attention(*inputs)


def test_ambiguous_causal_alignment_rejected():
    with pytest.raises(ValueError, match="equal aligned"):
        scaled_dot_product_attention(*random_inputs((2, 3), key_length=5), causal=True)


@pytest.mark.parametrize("mask_shape", [(4, 3), (3, 4), (1, 1, 3, 3)])
def test_mask_cannot_broadcast_to_weights(mask_shape):
    with pytest.raises(ValueError, match="broadcast"):
        scaled_dot_product_attention(*example(), mask=torch.zeros(mask_shape, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64])
def test_unsupported_input_dtypes(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        scaled_dot_product_attention(*example(dtype))


def test_mixed_input_dtype_rejected():
    q, k, v = example()
    with pytest.raises(TypeError, match="share"):
        scaled_dot_product_attention(q, k.float(), v)


def test_device_mismatch_without_claiming_gpu_verification():
    q, k, v = example()
    with pytest.raises(ValueError, match="share a device"):
        scaled_dot_product_attention(q, k.to("meta"), v)
    with pytest.raises(ValueError, match="mask must share"):
        scaled_dot_product_attention(
            q, k, v, mask=torch.zeros(3, 3, device="meta", dtype=torch.bool)
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_inputs_rejected(axis, bad):
    inputs = example()
    inputs[axis][0, 0] = bad
    with pytest.raises(ValueError, match="finite values"):
        scaled_dot_product_attention(*inputs)


def test_finite_inputs_with_overflowing_scores_rejected():
    huge = torch.full((2, 2), 1e30, dtype=torch.float32)
    with pytest.raises(ValueError, match="overflowed"):
        scaled_dot_product_attention(huge, huge, torch.ones_like(huge))


@pytest.mark.parametrize("bad", [1.0, -1.0, math.nan, math.inf])
def test_additive_mask_only_zero_or_negative_infinity(bad):
    mask = torch.zeros(3, 3, dtype=torch.float64)
    mask[0, 1] = bad
    with pytest.raises(ValueError, match="0 or -inf"):
        scaled_dot_product_attention(*example(), mask=mask)


@pytest.mark.parametrize(
    "mask", [torch.ones(3, 3, dtype=torch.int64), torch.zeros(3, 3, dtype=torch.float32)]
)
def test_incompatible_mask_dtype(mask):
    with pytest.raises(TypeError, match="mask must be bool"):
        scaled_dot_product_attention(*example(), mask=mask)


@pytest.mark.parametrize("causal,additive", [(False, False), (False, True), (True, False)])
def test_fully_blocked_rows_rejected(causal, additive):
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, :] = True if not causal else False
    mask[0, 0] = True
    if additive:
        mask = torch.zeros(3, 3, dtype=torch.float64).masked_fill(mask, -math.inf)
    with pytest.raises(ValueError, match="at least one allowed key"):
        scaled_dot_product_attention(*example(), causal=causal, mask=mask)


@pytest.mark.parametrize("length", [0, -1, True, 2.0, "3"])
def test_invalid_causal_length(length):
    with pytest.raises(ValueError, match="positive integer"):
        causal_forbidden_mask(length)


def test_non_tensor_and_non_boolean_api_errors():
    q, k, v = example()
    with pytest.raises(TypeError, match="must be tensors"):
        scaled_dot_product_attention(Q, k, v)
    with pytest.raises(TypeError, match="mask must be a tensor"):
        scaled_dot_product_attention(q, k, v, mask=[[False]])
    with pytest.raises(TypeError, match="causal must be bool"):
        scaled_dot_product_attention(q, k, v, causal=1)


def test_inputs_not_mutated():
    inputs = example()
    before = tuple(x.clone() for x in inputs)
    mask = causal_forbidden_mask(3)
    mask_before = mask.clone()
    scaled_dot_product_attention(*inputs, mask=mask)
    assert all(torch.equal(a, b) for a, b in zip(inputs, before, strict=True))
    assert torch.equal(mask, mask_before)
