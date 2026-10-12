"""Inspectable scaled dot-product attention; no learned projections or fused attention."""

import math

import torch
from torch import Tensor


def causal_forbidden_mask(length: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return (length, length) bool mask: True blocks a future key, diagonal allowed."""
    if type(length) is not int or length <= 0:
        raise ValueError("length must be a positive integer")
    return torch.ones((length, length), dtype=torch.bool, device=device).triu(diagonal=1)


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Return (output, weights) from explicit QK^T / sqrt(d_k), softmax and AV.

    Q (..., Tq, dk), K (..., Tk, dk), V (..., Tk, dv) have identical leading
    dimensions (including optional batch/head axes), dtype and device. Output is
    (..., Tq, dv); weights (..., Tq, Tk). Float32/float64, positive dimensions and
    finite inputs are supported. Leading input axes are not implicitly broadcast.
    Unmasked Tq != Tk is allowed; causal=True requires Tq == Tk with aligned indices.
    Mask broadcasts TO weights shape: bool True means forbidden, or same-dtype
    additive entries exclusively 0 (allowed) / -inf (forbidden). It combines with
    the causal mask by union. Fully blocked rows and nonfinite computed scores
    raise ValueError. No dropout, projections, dtype conversion or device fallback.
    """
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    if not all(isinstance(x, Tensor) for x in (q, k, v)):
        raise TypeError("q, k and v must be tensors")
    if any(x.ndim < 2 or any(dim <= 0 for dim in x.shape) for x in (q, k, v)):
        raise ValueError("q, k and v require at least two axes and positive dimensions")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k and v must have identical leading dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must share a device")
    if q.dtype not in (torch.float32, torch.float64) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must share float32 or float64 dtype")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires equal aligned query/key lengths")
    if not all(torch.isfinite(x).all().item() for x in (q, k, v)):
        raise ValueError("q, k and v must contain only finite values")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all().item():
        raise ValueError("computed attention scores must be finite; input magnitude overflowed")
    forbidden_mask = torch.zeros_like(scores, dtype=torch.bool)
    if mask is not None:
        if not isinstance(mask, Tensor):
            raise TypeError("mask must be a tensor")
        if mask.device != q.device:
            raise ValueError("mask must share the input device")
        if mask.dtype == torch.bool:
            supplied_forbidden = mask
        elif mask.dtype == q.dtype:
            if not ((mask == 0) | torch.isneginf(mask)).all().item():
                raise ValueError("additive mask entries must be 0 or -inf")
            supplied_forbidden = torch.isneginf(mask)
        else:
            raise TypeError("mask must be bool or the same floating dtype as inputs")
        try:
            forbidden_mask = torch.broadcast_to(supplied_forbidden, scores.shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to attention weights shape") from exc
    if causal:
        forbidden_mask = forbidden_mask | causal_forbidden_mask(q.shape[-2], device=q.device)
    if forbidden_mask.all(dim=-1).any().item():
        raise ValueError("every query row must have at least one allowed key")

    masked_scores = scores.masked_fill(forbidden_mask, -math.inf)
    weights = torch.softmax(masked_scores, dim=-1)
    output = torch.matmul(weights, v)
    if not torch.isfinite(output).all().item():
        raise ValueError("computed attention output must be finite")
    return output, weights
