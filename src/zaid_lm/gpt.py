"""Complete GPT causal language model: embeddings, decoder blocks, LM head."""

import math

import torch
from torch import nn

from zaid_lm.positional_encoding import SinusoidalPositionalEncoding
from zaid_lm.transformer_block import TransformerDecoderBlock


class GPT(nn.Module):
    """Decoder-only causal Transformer language model.

    Architecture (Pre-LN GPT-2 style):
        Token embedding -> Positional encoding -> N decoder blocks -> Final LN -> LM head

    The final LayerNorm is required in Pre-LN architecture because the last
    decoder block's output is not normalized (normalization happens at the
    beginning of each block, not after).

    Weight tying between token embedding and LM head is NOT used by default
    to keep the architecture transparent and independently testable. It can
    be enabled explicitly if desired.
    """

    def __init__(self, config):
        """Initialize from a ProjectConfig instance.

        Requires config.vocab_size to be a resolved positive integer
        (not None). All model dimensions come from the config.
        """
        super().__init__()
        # Validate essential config fields
        if config.vocab_size is None:
            raise ValueError(
                "vocab_size must be resolved before constructing the model; fit the tokenizer first"
            )
        if isinstance(config.vocab_size, bool) or not isinstance(config.vocab_size, int):
            raise ValueError("vocab_size must be a positive integer")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length

        # Token embedding: vocab_size -> d_model
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Sinusoidal positional encoding (zero trainable parameters)
        self.position_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.context_length, dropout=config.dropout
        )

        # N stacked decoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                    activation=config.activation,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final LayerNorm (required for Pre-LN architecture)
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head: d_model -> vocab_size (no bias, standard practice)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply sensible weight initialization.

        - Embedding: normal with std=0.02 (GPT-2 convention)
        - Linear layers: normal with std=0.02
        - LayerNorm: bias=0, weight=1 (PyTorch default, explicit for clarity)
        - Output projection scaling: std / sqrt(2 * n_layers) for residual paths
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale output projections in attention and FFN for residual depth
        # Each block has 2 residual paths, so scale by 1/sqrt(2*n_layers)
        residual_scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.mha.out_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(block.ffn.linear2.weight, mean=0.0, std=0.02 * residual_scale)

    def forward(self, input_ids, *, targets=None, return_attention=False):
        """Forward pass through the complete GPT model.

        Args:
            input_ids: (B, T) integer token IDs in [0, vocab_size).
            targets: Optional (B, T) integer target IDs for loss computation.
                     When provided, targets[i] is the next token after input_ids[i].
            return_attention: If True, collect attention weights from all blocks.

        Returns:
            If targets is None:
                logits (B, T, vocab_size), or (logits, all_attention_weights) if return_attention.
            If targets is provided:
                (logits, loss), or (logits, loss, all_attention_weights) if return_attention.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, T)")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be integer dtype (int32 or int64)")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length {self.context_length}"
            )
        if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
            raise ValueError(f"input_ids must be in [0, {self.vocab_size})")

        # Token embedding: (B, T) -> (B, T, d_model)
        x = self.token_embedding(input_ids)

        # Add positional encoding: (B, T, d_model) -> (B, T, d_model)
        x = self.position_encoding(x)

        # Pass through N decoder blocks
        all_attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn_weights = block(x, causal=True, return_attention=True)
                all_attention_weights.append(attn_weights)
            else:
                x = block(x, causal=True)

        # Final LayerNorm
        x = self.final_norm(x)

        # LM head: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                raise TypeError("targets must be a tensor")
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype not in (torch.int32, torch.int64):
                raise TypeError("targets must be integer dtype")
            if (targets < 0).any() or (targets >= self.vocab_size).any():
                raise ValueError(f"targets must be in [0, {self.vocab_size})")
            # Cross-entropy loss: flatten (B*T, vocab_size) vs (B*T,)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        # Build return value
        if targets is not None:
            if return_attention:
                return logits, loss, all_attention_weights
            return logits, loss
        if return_attention:
            return logits, all_attention_weights
        return logits

    def count_parameters(self, *, trainable_only=True):
        """Return total parameter count with optional trainable filter."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self):
        """Return a dict mapping component names to their parameter counts."""
        breakdown = {}
        breakdown["token_embedding"] = sum(p.numel() for p in self.token_embedding.parameters())
        breakdown["position_encoding"] = sum(p.numel() for p in self.position_encoding.parameters())
        for i, block in enumerate(self.blocks):
            block_params = {}
            block_params["ln1"] = sum(p.numel() for p in block.ln1.parameters())
            block_params["mha"] = sum(p.numel() for p in block.mha.parameters())
            block_params["ln2"] = sum(p.numel() for p in block.ln2.parameters())
            block_params["ffn"] = sum(p.numel() for p in block.ffn.parameters())
            block_params["total"] = sum(block_params.values())
            breakdown[f"block_{i}"] = block_params
        breakdown["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())
        breakdown["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())
        breakdown["total_trainable"] = self.count_parameters(trainable_only=True)
        breakdown["total_all"] = self.count_parameters(trainable_only=False)
        return breakdown

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate next tokens given conditioning prefix input_ids (B, T).

        Args:
            input_ids: (B, T) conditioning token IDs.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (> 0.0). Higher = more diverse, lower = more deterministic.
            top_k: Optional top-k filtering threshold.
            top_p: Optional nucleus sampling (top-p) threshold.
            repetition_penalty: Optional penalty (> 1.0) applied to previously generated tokens.

        Returns:
            (B, T + max_new_tokens) sequence of generated token IDs.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # Crop conditioning prefix to context_length if it exceeds max capacity
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.context_length
                else input_ids[:, -self.context_length :]
            )

            # Forward pass: (B, T_cond, vocab_size)
            logits = self(idx_cond)

            # Focus only on logits at the last position: (B, vocab_size)
            logits = logits[:, -1, :]

            # Apply repetition penalty if specified
            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    for token_id in set(input_ids[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # Temperature and sampling
            if temperature <= 0.0 or temperature < 1e-6:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("Inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)

        if was_training:
            self.train()

        return input_ids
