"""Positions followed by attention: a demonstration, not a decoder block."""

import io

import pytest
import torch

from zaid_lm.multihead_attention import MultiHeadSelfAttention
from zaid_lm.positional_encoding import SinusoidalPositionalEncoding
from zaid_lm.reproducibility import seed_everything


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def modules():
    return (
        SinusoidalPositionalEncoding(8, 12, dtype=torch.float64),
        MultiHeadSelfAttention(8, 2, dtype=torch.float64),
    )


def test_composition_shapes_finite_gradients_and_no_mutation():
    position, attention = modules()
    x = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    before = x.detach().clone()
    output, weights = attention(position(x), return_attention=True)
    assert output.shape == (2, 6, 8) and weights.shape == (2, 2, 6, 6)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    for parameter in attention.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
    assert list(position.parameters()) == []


def test_future_token_inputs_fixed_positions_preserve_prefix():
    position, attention = modules()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    original = attention(position(x, offset=2))
    for prefix in range(1, 7):
        future = x.clone()
        future[:, prefix:] += 11
        changed, weights = attention(position(future, offset=2), return_attention=True)
        torch.testing.assert_close(changed[:, :prefix], original[:, :prefix], rtol=0, atol=0)
        assert torch.count_nonzero(torch.triu(weights, 1)) == 0


def test_seeded_reproducibility_and_state_roundtrip():
    seed_everything(42, device="cpu")
    position, attention = modules()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    expected = attention(position(x))
    seed_everything(42, device="cpu")
    p2, a2 = modules()
    x2 = torch.randn(2, 5, 8, dtype=torch.float64)
    torch.testing.assert_close(a2(p2(x2)), expected, rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save({"positions": position.state_dict(), "attention": attention.state_dict()}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    p3, a3 = modules()
    p3.load_state_dict(restored["positions"])
    a3.load_state_dict(restored["attention"])
    torch.testing.assert_close(a3(p3(x)), expected, rtol=0, atol=0)


def test_positions_break_unmasked_permutation_symmetry():
    seed_everything(42, device="cpu")
    position, attention = modules()
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    order = torch.tensor([3, 0, 4, 1, 2])
    plain = attention(x, causal=False)
    torch.testing.assert_close(
        attention(x[:, order], causal=False), plain[:, order], rtol=1e-12, atol=1e-12
    )
    aware = attention(position(x), causal=False)
    # Tokens move, but absolute position indices remain assigned to their new slots.
    reordered = attention(position(x[:, order]), causal=False)
    assert (reordered - aware[:, order]).abs().max() > 1e-3
