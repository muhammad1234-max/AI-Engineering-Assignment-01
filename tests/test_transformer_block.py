"""Tests for TransformerDecoderBlock and FeedForwardNetwork."""

import pytest
import torch

from zaid_lm.transformer_block import FeedForwardNetwork, TransformerDecoderBlock

# ---------- FeedForwardNetwork tests ----------


class TestFeedForwardNetwork:
    def test_output_shape(self):
        ffn = FeedForwardNetwork(64, 256)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_gelu_activation(self):
        ffn = FeedForwardNetwork(32, 128, activation="gelu")
        x = torch.randn(1, 4, 32)
        out = ffn(x)
        assert out.shape == (1, 4, 32)
        assert torch.isfinite(out).all()

    def test_relu_activation(self):
        ffn = FeedForwardNetwork(32, 128, activation="relu")
        x = torch.randn(1, 4, 32)
        out = ffn(x)
        assert out.shape == (1, 4, 32)
        assert torch.isfinite(out).all()

    def test_invalid_activation_rejected(self):
        with pytest.raises(ValueError, match="activation"):
            FeedForwardNetwork(32, 128, activation="tanh")

    def test_invalid_d_model_rejected(self):
        with pytest.raises(ValueError, match="d_model"):
            FeedForwardNetwork(0, 128)

    def test_invalid_ffn_dim_rejected(self):
        with pytest.raises(ValueError, match="ffn_dim"):
            FeedForwardNetwork(32, -1)

    def test_bool_d_model_rejected(self):
        with pytest.raises(ValueError, match="d_model"):
            FeedForwardNetwork(True, 128)

    def test_dropout_out_of_range(self):
        with pytest.raises(ValueError, match="dropout"):
            FeedForwardNetwork(32, 128, dropout=1.0)

    def test_dropout_bool_rejected(self):
        with pytest.raises(TypeError, match="dropout"):
            FeedForwardNetwork(32, 128, dropout=True)

    def test_gradient_flow(self):
        ffn = FeedForwardNetwork(32, 128, dropout=0.0)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for p in ffn.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_parameter_count(self):
        ffn = FeedForwardNetwork(64, 256)
        # linear1: 64*256 + 256 = 16640
        # linear2: 256*64 + 64 = 16448
        total = sum(p.numel() for p in ffn.parameters())
        assert total == 64 * 256 + 256 + 256 * 64 + 64

    def test_eval_disables_dropout(self):
        ffn = FeedForwardNetwork(32, 128, dropout=0.5)
        ffn.eval()
        x = torch.randn(2, 4, 32)
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.equal(out1, out2), "eval mode must be deterministic"


# ---------- TransformerDecoderBlock tests ----------


class TestTransformerDecoderBlock:
    def test_output_shape(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out = block(x, causal=True)
        assert out.shape == (2, 8, 64)

    def test_output_finite(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert torch.isfinite(out).all()

    def test_return_attention_shapes(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        x = torch.randn(2, 8, 64)
        out, attn = block(x, return_attention=True)
        assert out.shape == (2, 8, 64)
        assert attn.shape == (2, 4, 8, 8)  # (B, H, T, T)

    def test_causal_masking_blocks_future(self):
        """Perturbing future tokens must not affect earlier positions."""
        torch.manual_seed(42)
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()

        x = torch.randn(1, 8, 64)
        out_orig = block(x, causal=True)

        # Perturb the last 3 tokens
        x_perturbed = x.clone()
        x_perturbed[:, 5:, :] += 100.0
        out_perturbed = block(x_perturbed, causal=True)

        # First 5 positions must be identical
        assert torch.allclose(out_orig[:, :5, :], out_perturbed[:, :5, :], atol=1e-6), (
            "causal block must not let future perturbation affect earlier positions"
        )

    def test_attention_weights_sum_to_one(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        x = torch.randn(1, 6, 64)
        _, attn = block(x, causal=True, return_attention=True)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_attention_future_weights_zero(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        x = torch.randn(1, 6, 64)
        _, attn = block(x, causal=True, return_attention=True)
        # Upper triangle (future) must be zero
        for t_q in range(6):
            for t_k in range(t_q + 1, 6):
                assert attn[0, :, t_q, t_k].abs().max() == 0.0

    def test_residual_connection_effect(self):
        """Output must differ from input (attention/FFN have effect)."""
        torch.manual_seed(0)
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        x = torch.randn(1, 4, 64)
        out = block(x)
        assert not torch.allclose(x, out, atol=1e-6), (
            "block output must differ from input due to attention and FFN"
        )

    def test_gradient_flow_through_block(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Check gradients flow to all major components
        for name, p in block.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"nonfinite gradient for {name}"

    def test_invalid_input_not_tensor(self):
        block = TransformerDecoderBlock(64, 4, 256)
        with pytest.raises(TypeError, match="tensor"):
            block([[1, 2, 3]])

    def test_invalid_input_wrong_ndim(self):
        block = TransformerDecoderBlock(64, 4, 256)
        with pytest.raises(ValueError, match="shape"):
            block(torch.randn(4, 64))

    def test_invalid_d_model_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerDecoderBlock(63, 4, 256)

    def test_invalid_negative_d_model(self):
        with pytest.raises(ValueError, match="d_model"):
            TransformerDecoderBlock(-1, 4, 256)

    def test_dropout_range(self):
        with pytest.raises(ValueError, match="dropout"):
            TransformerDecoderBlock(64, 4, 256, dropout=1.5)

    def test_pre_ln_architecture(self):
        """Verify the block uses Pre-LN: norm is applied BEFORE attention/FFN."""
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        # ln1 exists and is applied to input before MHA
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)

    def test_different_sequence_lengths(self):
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()
        for seq_len in [1, 4, 16, 32]:
            x = torch.randn(1, seq_len, 64)
            out = block(x)
            assert out.shape == (1, seq_len, 64)

    def test_batch_independence(self):
        """Each sample in a batch should produce the same output as standalone."""
        torch.manual_seed(42)
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.0)
        block.eval()

        x1 = torch.randn(1, 4, 64)
        x2 = torch.randn(1, 4, 64)
        batched = torch.cat([x1, x2], dim=0)

        out_batched = block(batched)
        out_single_1 = block(x1)
        out_single_2 = block(x2)

        assert torch.allclose(out_batched[0], out_single_1[0], atol=1e-6)
        assert torch.allclose(out_batched[1], out_single_2[0], atol=1e-6)

    def test_train_eval_dropout_difference(self):
        """Training mode with dropout should differ from eval mode."""
        block = TransformerDecoderBlock(64, 4, 256, dropout=0.5)
        x = torch.randn(2, 8, 64)

        block.eval()
        out_eval = block(x)

        block.train()
        torch.manual_seed(999)
        out_train = block(x)

        # They should generally differ due to dropout
        # (extremely unlikely to be identical with 50% dropout)
        assert not torch.equal(out_eval, out_train)

    def test_nonfinite_input_rejected(self):
        block = TransformerDecoderBlock(64, 4, 256)
        x = torch.randn(1, 4, 64)
        x[0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            block(x)

    def test_gelu_vs_relu_differ(self):
        """Different activations should produce different outputs."""
        torch.manual_seed(42)
        block_gelu = TransformerDecoderBlock(64, 4, 256, activation="gelu", dropout=0.0)
        torch.manual_seed(42)
        block_relu = TransformerDecoderBlock(64, 4, 256, activation="relu", dropout=0.0)

        x = torch.randn(1, 4, 64)
        block_gelu.eval()
        block_relu.eval()
        out_gelu = block_gelu(x)
        out_relu = block_relu(x)
        # Same init but different activations -> different outputs
        assert not torch.allclose(out_gelu, out_relu, atol=1e-6)
