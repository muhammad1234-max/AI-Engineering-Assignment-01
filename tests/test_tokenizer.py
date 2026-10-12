"""Unit tests for CharacterTokenizer: fitting, encoding, decoding, and persistence."""

from __future__ import annotations

import pytest
import torch

from zaid_lm.tokenizer import CharacterTokenizer


class TestCharacterTokenizer:
    def test_fit_from_text(self):
        text = "banana"
        tokenizer = CharacterTokenizer.from_text(text)
        assert tokenizer.vocab_size == 3  # 'a', 'b', 'n'
        assert tokenizer.vocabulary == ["a", "b", "n"]

    def test_encode_decode_roundtrip(self):
        text = "Hello, World! 123"
        tokenizer = CharacterTokenizer.from_text(text)
        ids = tokenizer.encode(text)
        assert len(ids) == len(text)
        recovered = tokenizer.decode(ids)
        assert recovered == text

    def test_encode_as_tensor(self):
        text = "abc"
        tokenizer = CharacterTokenizer.from_text(text)
        ids_tensor = tokenizer.encode(text, as_tensor=True)
        assert isinstance(ids_tensor, torch.Tensor)
        assert ids_tensor.dtype == torch.long
        assert ids_tensor.tolist() == tokenizer.encode(text, as_tensor=False)

    def test_decode_tensor_input(self):
        text = "test"
        tokenizer = CharacterTokenizer.from_text(text)
        ids = tokenizer.encode(text, as_tensor=True)
        assert tokenizer.decode(ids) == text
        # Also test 2D batch tensor of shape (1, T)
        assert tokenizer.decode(ids.unsqueeze(0)) == text

    def test_unknown_character_raises_without_unk(self):
        tokenizer = CharacterTokenizer.from_text("abc")
        with pytest.raises(KeyError, match="not in tokenizer vocabulary"):
            tokenizer.encode("abcd")

    def test_unknown_character_with_unk_token(self):
        tokenizer = CharacterTokenizer.from_text("abc", unk_token="<unk>")
        assert "<unk>" in tokenizer.vocabulary
        encoded = tokenizer.encode("abz", as_tensor=False)
        assert encoded[2] == tokenizer.unk_idx
        decoded = tokenizer.decode(encoded)
        assert decoded == "ab<unk>"

    def test_save_and_load(self, tmp_path):
        save_file = tmp_path / "tokenizer.json"
        text = "The quick brown fox jumps over the lazy dog."
        orig = CharacterTokenizer.from_text(text, unk_token="<?>")
        orig.save(save_file)

        loaded = CharacterTokenizer.load(save_file)
        assert loaded.vocab_size == orig.vocab_size
        assert loaded.vocabulary == orig.vocabulary
        assert loaded.unk_token == orig.unk_token
        assert loaded.unk_idx == orig.unk_idx

        # Check encoding parity
        test_str = "quick dog <?>"
        assert orig.encode(test_str) == loaded.encode(test_str)

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="Cannot fit tokenizer on empty text"):
            CharacterTokenizer.from_text("")

    def test_uninitialized_tokenizer_rejected(self):
        tok = CharacterTokenizer()
        with pytest.raises(RuntimeError, match="not initialized"):
            tok.encode("a")
        with pytest.raises(RuntimeError, match="not initialized"):
            tok.decode([0])
