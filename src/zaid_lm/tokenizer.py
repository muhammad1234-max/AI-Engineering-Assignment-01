"""Deterministic character-level tokenizer for causal language modeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch


class CharacterTokenizer:
    """Character-level tokenizer with deterministic ordering and explicit vocabulary.

    The vocabulary is strictly fitted on the training split to avoid data leakage.
    Supports optional unknown token handling, save/load, and bidirectional mapping.
    """

    def __init__(
        self,
        vocabulary: Sequence[str] | None = None,
        *,
        unk_token: str | None = None,
    ) -> None:
        self.unk_token = unk_token
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.unk_idx: int | None = None

        if vocabulary is not None:
            self._build_vocabulary(vocabulary)

    def _build_vocabulary(self, characters: Sequence[str]) -> None:
        """Build character-to-index and index-to-character mappings."""
        # Preserve order while ensuring uniqueness
        seen: set[str] = set()
        ordered_chars: list[str] = []
        for ch in characters:
            if ch not in seen:
                seen.add(ch)
                ordered_chars.append(ch)

        if self.unk_token is not None and self.unk_token not in seen:
            ordered_chars.append(self.unk_token)

        self.char2idx = {ch: idx for idx, ch in enumerate(ordered_chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(ordered_chars)}

        if self.unk_token is not None:
            self.unk_idx = self.char2idx[self.unk_token]
        else:
            self.unk_idx = None

    @classmethod
    def from_text(cls, text: str, *, unk_token: str | None = None) -> CharacterTokenizer:
        """Fit tokenizer vocabulary exclusively on the provided training text."""
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text")
        unique_chars = sorted(list(set(text)))
        tokenizer = cls(vocabulary=unique_chars, unk_token=unk_token)
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    @property
    def vocabulary(self) -> list[str]:
        return [self.idx2char[i] for i in range(len(self.idx2char))]

    def encode(self, text: str, *, as_tensor: bool = False) -> list[int] | torch.Tensor:
        """Encode a string into a list or 1D LongTensor of token IDs."""
        if not self.char2idx:
            raise RuntimeError("Tokenizer vocabulary is not initialized")

        ids: list[int] = []
        for ch in text:
            if ch in self.char2idx:
                ids.append(self.char2idx[ch])
            elif self.unk_idx is not None:
                ids.append(self.unk_idx)
            else:
                raise KeyError(
                    f"Character {repr(ch)} not in tokenizer vocabulary and no unk_token configured"
                )

        if as_tensor:
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        """Decode a sequence of integer token IDs back into a string."""
        if not self.idx2char:
            raise RuntimeError("Tokenizer vocabulary is not initialized")

        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()

        chars: list[str] = []
        for idx in token_ids:
            if not isinstance(idx, int):
                idx = int(idx)
            if idx not in self.idx2char:
                raise KeyError(f"Token ID {idx} out of vocabulary range [0, {self.vocab_size})")
            chars.append(self.idx2char[idx])

        return "".join(chars)

    def save(self, path: Path | str) -> None:
        """Save tokenizer configuration and vocabulary mapping to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "unk_idx": self.unk_idx,
            "vocabulary": self.vocabulary,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> CharacterTokenizer:
        """Load tokenizer from saved JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls(vocabulary=data["vocabulary"], unk_token=data.get("unk_token"))
        return tokenizer
