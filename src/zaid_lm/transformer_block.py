"""Transformer decoder block: Pre-LN attention and FFN with residual paths."""

import math

import torch
from torch import nn

from zaid_lm.multihead_attention import MultiHeadSelfAttention


class FeedForwardNetwork(nn.Module):
    """Two-linear-layer network with nonlinearity: d_model -> ffn_dim -> d_model.

    GELU is the recommended activation. ReLU is also supported.
    Dropout is applied after the second linear layer.
    """

    def __init__(
        self, d_model, ffn_dim, activation="gelu", dropout=0.0, *, device=None, dtype=None
    ):
        super().__init__()
        for name, value in (("d_model", d_model), ("ffn_dim", ffn_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a finite number")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        options = {"device": device, "dtype": dtype}
        self.linear1 = nn.Linear(d_model, ffn_dim, **options)
        self.linear2 = nn.Linear(ffn_dim, d_model, **options)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerDecoderBlock(nn.Module):
    """Pre-LN decoder block: norm -> attention + residual, norm -> FFN + residual.

    Pre-LN architecture normalizes BEFORE each sublayer, which produces more
    stable gradients and is the standard approach in GPT-2 and later models.
    This contrasts with Post-LN (original Transformer) which normalizes after
    the residual addition.

    Sublayer 1:  x + Dropout(MHA(LayerNorm(x)))
    Sublayer 2:  x + Dropout(FFN(LayerNorm(x)))

    The MHA module already includes output projection dropout, so the residual
    dropout here is an additional regularization layer around each sublayer path.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ffn_dim,
        activation="gelu",
        dropout=0.0,
        bias=True,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads), ("ffn_dim", ffn_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a finite number")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        options = {"device": device, "dtype": dtype}

        # Sublayer 1: Multi-Head Self-Attention
        self.ln1 = nn.LayerNorm(d_model, **options)
        self.mha = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, bias=bias, **options)

        # Sublayer 2: Feed-Forward Network
        self.ln2 = nn.LayerNorm(d_model, **options)
        self.ffn = FeedForwardNetwork(
            d_model, ffn_dim, activation=activation, dropout=dropout, **options
        )

        # Residual dropout (separate from sublayer-internal dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, *, causal=True, mask=None, return_attention=False):
        """Process (B, T, d_model) through attention + FFN with residual paths.

        Returns output tensor, or (output, attention_weights) if return_attention=True.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim != 3 or any(size <= 0 for size in x.shape):
            raise ValueError("x must have nonempty shape (B, T, d_model)")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")

        # Sublayer 1: Pre-LN -> MHA -> residual
        normed1 = self.ln1(x)
        attn_result = self.mha(normed1, causal=causal, mask=mask, return_attention=return_attention)
        if return_attention:
            attn_output, attn_weights = attn_result
        else:
            attn_output = attn_result
            attn_weights = None
        x = x + self.residual_dropout(attn_output)

        # Sublayer 2: Pre-LN -> FFN -> residual
        normed2 = self.ln2(x)
        ffn_output = self.ffn(normed2)
        x = x + self.residual_dropout(ffn_output)

        if not torch.isfinite(x).all():
            raise ValueError("decoder block produced nonfinite values")

        if return_attention:
            return x, attn_weights
        return x
