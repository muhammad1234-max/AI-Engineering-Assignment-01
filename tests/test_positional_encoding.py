"""Independent sinusoidal mathematics, state, boundaries and input contracts."""

import io
import math

import pytest
import torch

from experiments.visualize_positions import scalar_positions
from zaid_lm.config import load_config
from zaid_lm.positional_encoding import SinusoidalPositionalEncoding


@pytest.mark.parametrize("width", [1, 3, 4, 7, 128])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_every_feature_matches_independent_scalar_reference(width, dtype, tolerance):
    module = SinusoidalPositionalEncoding(width, 128, dtype=dtype)
    actual = module(torch.zeros(1, 128, width, dtype=dtype))
    expected = torch.tensor(scalar_positions(128, width), dtype=torch.float64).unsqueeze(0)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=tolerance)
    assert actual.shape == (1, 128, width)
    torch.testing.assert_close(
        actual[0, 0, 0::2], torch.zeros_like(actual[0, 0, 0::2]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual[0, 0, 1::2], torch.ones_like(actual[0, 0, 1::2]), rtol=0, atol=0
    )
    assert not torch.equal(actual[:, 0], actual[:, 1])
    # Pair identity is an additional independent structural check.
    for even in range(0, width - 1, 2):
        torch.testing.assert_close(
            actual[0, :, even] ** 2 + actual[0, :, even + 1] ** 2,
            torch.ones(128, dtype=dtype),
            rtol=0,
            atol=tolerance,
        )


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_addition_broadcast_and_no_mutation(batch):
    module = SinusoidalPositionalEncoding(4, 8, dtype=torch.float64).eval()
    x = torch.arange(batch * 12, dtype=torch.float64).reshape(batch, 3, 4)
    original = x.clone()
    y = module(x)
    assert y.shape == x.shape
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(y - x, module.encoding[:, :3].expand_as(x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module(x), y, rtol=0, atol=0)


def test_actual_baseline():
    config = load_config("configs/baseline.json")
    module = SinusoidalPositionalEncoding(
        config.d_model, config.context_length, config.dropout
    ).eval()
    output = module(torch.zeros(2, 128, 128))
    assert module.encoding.shape == (1, 128, 128)
    assert output.shape == (2, 128, 128) and torch.isfinite(output).all()
    assert config.vocab_size is None and config.tokenizer == "character"
    assert output[0, 1, 0].item() == pytest.approx(math.sin(1), abs=1e-6)
    assert output[0, 1, 1].item() == pytest.approx(math.cos(1), abs=1e-6)


def test_context_boundary_and_absolute_offset():
    module = SinusoidalPositionalEncoding(5, 8, dtype=torch.float64)
    whole = module(torch.zeros(1, 8, 5, dtype=torch.float64))
    final = module(torch.zeros(2, 3, 5, dtype=torch.float64), offset=5)
    torch.testing.assert_close(final, whole[:, 5:].expand(2, -1, -1), rtol=0, atol=0)
    with pytest.raises(ValueError, match="context_length"):
        module(torch.zeros(1, 9, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="context_length"):
        module(torch.zeros(1, 4, 5, dtype=torch.float64), offset=5)


@pytest.mark.parametrize("offset", [-1, True, 0.5, "0", None])
def test_invalid_offset(offset):
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(4, 8)(torch.zeros(1, 2, 4), offset=offset)


@pytest.mark.parametrize(
    "width,context", [(0, 8), (-2, 8), (True, 8), (4.0, 8), (4, 0), (4, -1), (4, True), (4, 8.0)]
)
def test_invalid_dimensions(width, context):
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(width, context)


@pytest.mark.parametrize("shape", [(3, 4), (1, 2, 3, 4), (0, 2, 4), (1, 0, 4), (1, 2, 3)])
def test_invalid_input_shape(shape):
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(4, 8)(torch.zeros(shape))


def test_registered_persistent_buffer_no_parameters_and_serialization():
    module = SinusoidalPositionalEncoding(7, 9, dtype=torch.float64)
    assert list(module.parameters()) == []
    assert dict(module.named_buffers())["encoding"] is module.encoding
    assert not module.encoding.requires_grad
    assert list(module.state_dict()) == ["encoding"]
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    restored = SinusoidalPositionalEncoding(7, 9, dtype=torch.float64)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    x = torch.randn(2, 3, 7, dtype=torch.float64)
    torch.testing.assert_close(restored(x, offset=2), module(x, offset=2), rtol=0, atol=0)


def test_dtype_device_contract_and_buffer_migration_cpu_only():
    module = SinusoidalPositionalEncoding(4, 8)
    with pytest.raises(ValueError, match="dtype and device"):
        module(torch.zeros(1, 2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype and device"):
        module(torch.empty(1, 2, 4, device="meta"))
    original = module.encoding.clone()
    module.double()
    torch.testing.assert_close(module.encoding, original.double(), rtol=0, atol=0)
    assert module(torch.zeros(1, 2, 4, dtype=torch.float64)).dtype == torch.float64
    module.to(device="meta")
    assert module.encoding.device.type == "meta"


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_invalid_input_dtype(dtype):
    with pytest.raises(TypeError):
        SinusoidalPositionalEncoding(4, 8)(torch.zeros(1, 2, 4, dtype=dtype))


def test_invalid_tensor_and_constructor_dtype():
    with pytest.raises(TypeError):
        SinusoidalPositionalEncoding(4, 8)(None)
    with pytest.raises(TypeError):
        SinusoidalPositionalEncoding(4, 8, dtype=torch.float16)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_input(value):
    with pytest.raises(ValueError, match="finite"):
        SinusoidalPositionalEncoding(4, 8)(torch.full((1, 2, 4), value))


def test_dropout_and_evaluation():
    module = SinusoidalPositionalEncoding(16, 16, dropout=0.5).eval()
    x = torch.ones(2, 16, 16)
    expected = module(x)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    module.train()
    y1, y2 = module(x), module(x)
    assert not torch.equal(y1, y2) and (y1 == 0).any()
    assert torch.all((y1 == 0) | torch.isclose(y1, expected * 2))
    module.eval()
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [-1, 1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_dropout(dropout):
    with pytest.raises((TypeError, ValueError)):
        SinusoidalPositionalEncoding(4, 8, dropout=dropout)


def test_addition_gradient_is_identity():
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    SinusoidalPositionalEncoding(5, 8, dtype=torch.float64)(x, offset=2).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
