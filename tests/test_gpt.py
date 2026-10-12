"""Tests for the complete GPT causal language model."""

import math
from dataclasses import replace

import pytest
import torch

from zaid_lm.config import ProjectConfig
from zaid_lm.gpt import GPT


def make_config(**overrides):
    """Create a small test config with resolved vocab_size."""
    defaults = {
        "vocab_size": 50,
        "context_length": 32,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.0,
        "seed": 42,
    }
    defaults.update(overrides)
    return ProjectConfig(**defaults)


class TestGPTConstruction:
    def test_basic_construction(self):
        config = make_config()
        model = GPT(config)
        assert model.vocab_size == 50
        assert model.d_model == 64
        assert model.context_length == 32

    def test_unresolved_vocab_rejected(self):
        config = make_config(vocab_size=None)
        # Need to bypass normal validation
        config = replace(config, vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size"):
            GPT(config)

    def test_zero_vocab_rejected(self):
        with pytest.raises(ValueError, match="vocab_size"):
            make_config(vocab_size=0)

    def test_head_divisibility_enforced(self):
        with pytest.raises(ValueError, match="divisible"):
            make_config(d_model=63, n_heads=4)


class TestGPTForward:
    def test_logits_shape(self):
        config = make_config()
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (2, 16))
        logits = model(ids)
        assert logits.shape == (2, 16, 50)

    def test_logits_finite(self):
        config = make_config()
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (1, 8))
        logits = model(ids)
        assert torch.isfinite(logits).all()

    def test_loss_computation(self):
        config = make_config()
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (2, 16))
        targets = torch.randint(0, 50, (2, 16))
        logits, loss = model(ids, targets=targets)
        assert logits.shape == (2, 16, 50)
        assert loss.ndim == 0  # scalar
        assert torch.isfinite(loss)
        assert loss > 0  # cross-entropy is positive

    def test_loss_near_log_vocab_at_init(self):
        """At random init, loss should be ~ln(vocab_size)."""
        config = make_config(vocab_size=50)
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (8, 16))
        targets = torch.randint(0, 50, (8, 16))
        _, loss = model(ids, targets=targets)
        expected = math.log(50)  # ~3.91
        assert abs(loss.item() - expected) < 1.5, (
            f"initial loss {loss.item():.2f} far from expected ~{expected:.2f}"
        )

    def test_return_attention_shapes(self):
        config = make_config(n_layers=3, n_heads=4)
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (1, 8))
        logits, attn_list = model(ids, return_attention=True)
        assert logits.shape == (1, 8, 50)
        assert len(attn_list) == 3  # one per layer
        for attn in attn_list:
            assert attn.shape == (1, 4, 8, 8)  # (B, H, T, T)

    def test_return_attention_with_loss(self):
        config = make_config(n_layers=2)
        model = GPT(config)
        model.eval()
        ids = torch.randint(0, 50, (1, 8))
        targets = torch.randint(0, 50, (1, 8))
        logits, loss, attn_list = model(ids, targets=targets, return_attention=True)
        assert logits.shape == (1, 8, 50)
        assert torch.isfinite(loss)
        assert len(attn_list) == 2


class TestGPTCausality:
    def test_future_perturbation_invariance(self):
        """Perturbing future token IDs must not affect earlier logits."""
        config = make_config(dropout=0.0)
        model = GPT(config)
        model.eval()

        ids = torch.randint(0, 50, (1, 16))
        logits_orig = model(ids)

        # Perturb the last 4 tokens
        ids_perturbed = ids.clone()
        ids_perturbed[:, 12:] = (ids_perturbed[:, 12:] + 25) % 50
        logits_perturbed = model(ids_perturbed)

        # First 12 positions must produce identical logits
        assert torch.allclose(logits_orig[:, :12, :], logits_perturbed[:, :12, :], atol=1e-5), (
            "GPT must be strictly causal — future tokens cannot affect earlier positions"
        )

    def test_attention_weights_causal(self):
        """All attention weights for future keys must be zero."""
        config = make_config(n_layers=2, n_heads=4, dropout=0.0)
        model = GPT(config)
        model.eval()

        ids = torch.randint(0, 50, (1, 8))
        _, attn_list = model(ids, return_attention=True)

        for layer_idx, attn in enumerate(attn_list):
            for t_q in range(8):
                for t_k in range(t_q + 1, 8):
                    max_future = attn[0, :, t_q, t_k].abs().max().item()
                    assert max_future == 0.0, (
                        f"layer {layer_idx}: query {t_q} attends to future key {t_k}"
                    )


class TestGPTGradients:
    def test_gradient_flow_to_all_parameters(self):
        config = make_config(dropout=0.0)
        model = GPT(config)
        ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        _, loss = model(ids, targets=targets)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no gradient for {name}"
                assert torch.isfinite(p.grad).all(), f"nonfinite gradient for {name}"

    def test_embedding_gradients(self):
        config = make_config()
        model = GPT(config)
        ids = torch.randint(0, 50, (1, 8))
        targets = torch.randint(0, 50, (1, 8))
        _, loss = model(ids, targets=targets)
        loss.backward()
        assert model.token_embedding.weight.grad is not None
        # Only used token IDs should have nonzero gradients
        used_ids = ids.unique()
        for token_id in used_ids:
            assert model.token_embedding.weight.grad[token_id].abs().sum() > 0


class TestGPTParameterCounting:
    def test_parameter_breakdown_sums(self):
        config = make_config()
        model = GPT(config)
        breakdown = model.parameter_breakdown()
        component_sum = (
            breakdown["token_embedding"]
            + breakdown["position_encoding"]
            + sum(breakdown[f"block_{i}"]["total"] for i in range(config.n_layers))
            + breakdown["final_norm"]
            + breakdown["lm_head"]
        )
        assert component_sum == breakdown["total_trainable"]

    def test_count_parameters_matches_sum(self):
        config = make_config()
        model = GPT(config)
        manual_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.count_parameters() == manual_sum

    def test_position_encoding_zero_trainable(self):
        config = make_config()
        model = GPT(config)
        breakdown = model.parameter_breakdown()
        assert breakdown["position_encoding"] == 0, (
            "sinusoidal encoding must have zero trainable parameters"
        )

    def test_expected_parameter_formula(self):
        """Check parameter count against analytical formula for baseline config."""
        config = make_config(vocab_size=50, d_model=64, n_heads=4, n_layers=2, ffn_dim=128)
        model = GPT(config)
        breakdown = model.parameter_breakdown()

        # Token embedding: vocab * d_model = 50 * 64 = 3200
        assert breakdown["token_embedding"] == 50 * 64

        # Per block:
        #   ln1: d_model + d_model = 128 (weight + bias)
        #   mha: 4 * (d_model^2 + d_model) = 4 * (4096 + 64) = 16640
        #   ln2: d_model + d_model = 128
        #   ffn: d_model*ffn + ffn + ffn*d_model + d_model = 64*128+128 + 128*64+64 = 16576
        per_block = breakdown["block_0"]["total"]
        assert breakdown["block_0"]["ln1"] == 64 * 2
        assert breakdown["block_0"]["mha"] == 4 * (64 * 64 + 64)
        assert breakdown["block_0"]["ln2"] == 64 * 2
        assert breakdown["block_0"]["ffn"] == 64 * 128 + 128 + 128 * 64 + 64
        assert per_block == (64 * 2) + 4 * (64 * 64 + 64) + (64 * 2) + (
            64 * 128 + 128 + 128 * 64 + 64
        )

        # Final norm: 2 * d_model = 128
        assert breakdown["final_norm"] == 64 * 2

        # LM head: d_model * vocab = 64 * 50 = 3200 (no bias)
        assert breakdown["lm_head"] == 64 * 50


class TestGPTInputValidation:
    def test_non_tensor_rejected(self):
        model = GPT(make_config())
        with pytest.raises(TypeError, match="tensor"):
            model([1, 2, 3])

    def test_wrong_ndim_rejected(self):
        model = GPT(make_config())
        with pytest.raises(ValueError, match="shape"):
            model(torch.tensor([1, 2, 3]))

    def test_float_dtype_rejected(self):
        model = GPT(make_config())
        with pytest.raises(TypeError, match="integer"):
            model(torch.randn(1, 4))

    def test_out_of_range_ids_rejected(self):
        model = GPT(make_config(vocab_size=50))
        with pytest.raises(ValueError, match=r"\[0, 50\)"):
            model(torch.tensor([[60, 1, 2, 3]]))

    def test_negative_ids_rejected(self):
        model = GPT(make_config(vocab_size=50))
        with pytest.raises(ValueError, match=r"\[0, 50\)"):
            model(torch.tensor([[-1, 1, 2, 3]]))

    def test_exceeds_context_rejected(self):
        model = GPT(make_config(context_length=8))
        ids = torch.randint(0, 50, (1, 16))
        with pytest.raises(ValueError, match="context_length"):
            model(ids)

    def test_targets_shape_mismatch_rejected(self):
        model = GPT(make_config())
        ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 4))
        with pytest.raises(ValueError, match="shape"):
            model(ids, targets=targets)

    def test_targets_float_rejected(self):
        model = GPT(make_config())
        ids = torch.randint(0, 50, (1, 4))
        with pytest.raises(TypeError, match="integer"):
            model(ids, targets=torch.randn(1, 4))


class TestGPTSequenceLengths:
    def test_single_token(self):
        model = GPT(make_config())
        model.eval()
        ids = torch.randint(0, 50, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, 50)

    def test_full_context(self):
        model = GPT(make_config(context_length=16))
        model.eval()
        ids = torch.randint(0, 50, (1, 16))
        logits = model(ids)
        assert logits.shape == (1, 16, 50)

    def test_various_lengths(self):
        model = GPT(make_config(context_length=32))
        model.eval()
        for length in [1, 2, 7, 15, 32]:
            ids = torch.randint(0, 50, (1, length))
            logits = model(ids)
            assert logits.shape == (1, length, 50)


class TestGPTTrainEval:
    def test_eval_deterministic(self):
        model = GPT(make_config(dropout=0.3))
        model.eval()
        ids = torch.randint(0, 50, (1, 8))
        out1 = model(ids)
        out2 = model(ids)
        assert torch.equal(out1, out2)

    def test_train_dropout_varies(self):
        model = GPT(make_config(dropout=0.5))
        model.train()
        ids = torch.randint(0, 50, (2, 8))
        torch.manual_seed(1)
        out1 = model(ids)
        torch.manual_seed(2)
        out2 = model(ids)
        assert not torch.equal(out1, out2), "training with 50% dropout must vary"

    def test_state_dict_roundtrip(self):
        config = make_config()
        model1 = GPT(config)
        model1.eval()
        ids = torch.randint(0, 50, (1, 8))
        out1 = model1(ids)

        # Save and reload
        state = model1.state_dict()
        model2 = GPT(config)
        model2.load_state_dict(state)
        model2.eval()
        out2 = model2(ids)

        assert torch.equal(out1, out2), "reloaded model must produce identical output"
