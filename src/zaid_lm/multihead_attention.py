"""Explicit head splitting and merging around the verified Phase 2 engine."""

import math

import torch
from torch import nn

from zaid_lm.attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """Self-attention on (batch, tokens, features); dropout follows output projection.

    Boolean masks forbid True entries; additive masks contain only 0 or -inf.
    Masks broadcast to (B, H, T, T); use (B, 1, T, T) for batch-specific masks.
    Inspection returns normalized, undropped per-head attention probabilities.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, *, device=None, dtype=None):
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a finite number")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(bias, bool):
            raise TypeError("bias must be bool")
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise TypeError("only float32 and float64 are supported")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        options = {"bias": bias, "device": device, "dtype": dtype}
        self.q_proj = nn.Linear(d_model, d_model, **options)
        self.k_proj = nn.Linear(d_model, d_model, **options)
        self.v_proj = nn.Linear(d_model, d_model, **options)
        self.out_proj = nn.Linear(d_model, d_model, **options)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, projected):
        batch, tokens, _ = projected.shape
        return projected.reshape(batch, tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def _merge_heads(self, heads):
        batch, _, tokens, _ = heads.shape
        return heads.permute(0, 2, 1, 3).contiguous().reshape(batch, tokens, self.d_model)

    def forward(self, x, *, causal=True, mask=None, return_attention=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim != 3 or any(size <= 0 for size in x.shape):
            raise ValueError("x must have nonempty shape (B, T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError("input embedding width must equal d_model")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("x must be float32 or float64")
        if x.dtype != self.q_proj.weight.dtype or x.device != self.q_proj.weight.device:
            raise ValueError("input and module must have the same dtype and device")
        if not isinstance(return_attention, bool):
            raise TypeError("return_attention must be bool")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        heads, weights = scaled_dot_product_attention(q, k, v, causal=causal, mask=mask)
        output = self.dropout(self.out_proj(self._merge_heads(heads)))
        if not torch.isfinite(output).all():
            raise ValueError("output projection or dropout produced nonfinite values")
        return (output, weights) if return_attention else output
