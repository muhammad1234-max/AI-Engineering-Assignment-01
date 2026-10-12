"""Unit tests for atomic checkpoint saving, loading, and parity verification."""

from __future__ import annotations

import pytest
import torch

from zaid_lm.checkpoint import load_checkpoint, save_checkpoint
from zaid_lm.config import ProjectConfig
from zaid_lm.gpt import GPT
from zaid_lm.tokenizer import CharacterTokenizer


class TestCheckpoint:
    def test_save_and_reload_weights_parity(self, tmp_path):
        ckpt_file = tmp_path / "model_ckpt.pt"

        tokenizer = CharacterTokenizer.from_text("abcdef")
        config = ProjectConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            ffn_dim=64,
        )

        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save checkpoint
        save_checkpoint(
            ckpt_file,
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=42,
            train_loss=1.234,
            val_loss=1.567,
        )
        assert ckpt_file.exists()

        # Load into fresh instance
        restored = load_checkpoint(ckpt_file)
        new_model = restored["model"]
        new_config = restored["config"]
        new_tokenizer = restored["tokenizer"]

        assert restored["step"] == 42
        assert restored["train_loss"] == 1.234
        assert restored["val_loss"] == 1.567
        assert new_config == config
        assert new_tokenizer.vocabulary == tokenizer.vocabulary

        # Test forward output parity
        test_ids = torch.tensor([[0, 1, 2, 3]])
        model.eval()
        new_model.eval()
        with torch.no_grad():
            out_orig = model(test_ids)
            out_restored = new_model(test_ids)

        assert torch.allclose(out_orig, out_restored, atol=1e-7)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_checkpoint(tmp_path / "non_existent.pt")
