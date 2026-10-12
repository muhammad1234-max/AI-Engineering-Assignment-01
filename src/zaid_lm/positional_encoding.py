"""Fixed sinusoidal position vectors, added once to token representations."""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add a persistent (1, context_length, d_model) buffer, then apply dropout.

    Input dtype/device must match the buffer. Construct with dtype=float64 for
    full double precision; .to() casts stored values using normal PyTorch rules.
    Offset selects absolute positions without introducing any cache machinery.
    Odd widths retain the final sine feature without an unmatched cosine.
    """

    def __init__(self, d_model, context_length, dropout=0.0, *, device=None, dtype=torch.float32):
        super().__init__()
        for name, value in (("d_model", d_model), ("context_length", context_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a finite number")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("only float32 and float64 are supported")
        self.d_model, self.context_length = d_model, context_length
        # Compute directly from the specified powers, in float64 before storage.
        even = torch.arange(0, d_model, 2, dtype=torch.float64, device=device)
        denominator = torch.pow(10000.0, even / d_model)
        positions = torch.arange(context_length, dtype=torch.float64, device=device)[:, None]
        angles = positions / denominator
        table = torch.empty(context_length, d_model, dtype=torch.float64, device=device)
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles[:, : d_model // 2])
        self.register_buffer("encoding", table.to(dtype=dtype).unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *, offset=0):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim != 3 or any(size <= 0 for size in x.shape):
            raise ValueError("x must have nonempty shape (B, T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError("input embedding width must equal d_model")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if offset + x.shape[1] > self.context_length:
            raise ValueError("offset plus sequence length exceeds context_length")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("x must be float32 or float64")
        if x.dtype != self.encoding.dtype or x.device != self.encoding.device:
            raise ValueError("input and positional buffer must have the same dtype and device")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        result = self.dropout(x + self.encoding[:, offset : offset + x.shape[1]])
        if not torch.isfinite(result).all():
            raise ValueError("position addition or dropout produced nonfinite values")
        return result
