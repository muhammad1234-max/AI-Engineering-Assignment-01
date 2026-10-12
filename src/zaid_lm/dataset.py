"""Dataset preparation, verification, and windowed sequence loading for Tiny Shakespeare."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import Dataset

from zaid_lm.tokenizer import CharacterTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
EXPECTED_BYTE_COUNT = 1115394


class DatasetSplits(NamedTuple):
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    tokenizer: CharacterTokenizer
    provenance: dict


def download_tiny_shakespeare(
    dest_path: Path | str,
    *,
    expected_sha256: str = TINY_SHAKESPEARE_SHA256,
) -> Path:
    """Download Tiny Shakespeare text and verify its cryptographic SHA256 digest."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        content = dest_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest == expected_sha256:
            return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        TINY_SHAKESPEARE_URL,
        headers={"User-Agent": "ZaidLM-Educational-Agent/1.0"},
    )
    with urllib.request.urlopen(req) as response:
        content = response.read()

    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for downloaded data. Expected {expected_sha256}, got {digest}"
        )

    dest_path.write_bytes(content)
    return dest_path


def prepare_dataset(
    data_dir: Path | str,
    *,
    train_ratio: float = 0.9,
    raw_filename: str = "input.txt",
    unk_token: str | None = None,
) -> DatasetSplits:
    """Download, split deterministically, fit tokenizer on train-only, and save provenance.

    Args:
        data_dir: Directory to store raw data and provenance.
        train_ratio: Fraction of tokens for training (e.g. 0.9 for 90/10 split).
        raw_filename: File name for raw text input.
        unk_token: Optional special token for unseen characters.

    Returns:
        DatasetSplits with (train_tokens, val_tokens, tokenizer, provenance).
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / raw_filename

    # Ensure downloaded and verified
    download_tiny_shakespeare(raw_path)

    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")
    line_count = text.count("\n")

    # Deterministic split without leakage
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Fit tokenizer strictly on training split
    tokenizer = CharacterTokenizer.from_text(train_text, unk_token=unk_token)

    # Encode splits
    train_tokens = tokenizer.encode(train_text, as_tensor=True)
    val_tokens = tokenizer.encode(val_text, as_tensor=True)

    # Check for any vocabulary leakage / unseen characters in validation
    val_unique = set(val_text)
    train_unique = set(train_text)
    unseen_chars = sorted(list(val_unique - train_unique))

    provenance = {
        "dataset_name": "Tiny Shakespeare",
        "source_url": TINY_SHAKESPEARE_URL,
        "raw_sha256": raw_sha256,
        "total_characters": len(text),
        "total_bytes": len(raw_bytes),
        "total_lines": line_count,
        "train_characters": len(train_text),
        "val_characters": len(val_text),
        "train_ratio": train_ratio,
        "vocab_size": tokenizer.vocab_size,
        "vocabulary": tokenizer.vocabulary,
        "unseen_chars_in_val": unseen_chars,
        "license": "Public Domain / CC0 (Shakespeare corpus)",
    }

    # Save tokenizer and provenance
    tokenizer.save(data_dir / "tokenizer.json")
    provenance_path = data_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    return DatasetSplits(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer=tokenizer,
        provenance=provenance,
    )


class ShakespeareDataset(Dataset):
    """PyTorch Dataset yielding windowed (x, y) token pairs for next-token prediction.

    x has shape (context_length,) containing tokens[i : i + context_length]
    y has shape (context_length,) containing tokens[i + 1 : i + context_length + 1]
    y[t] is the exact target for x[t].
    """

    def __init__(self, tokens: torch.Tensor, context_length: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D tensor")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(tokens) <= context_length:
            raise ValueError(
                f"tokens length ({len(tokens)}) must be greater than context_length ({context_length})"
            )

        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]
        return x, y


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (B, T) inputs and shifted (B, T) next-token targets.

    Args:
        data: 1D LongTensor of token IDs.
        batch_size: Number of sequences in batch (B).
        context_length: Sequence length per sample (T).
        device: Target execution device ('cpu' or 'cuda').
        generator: Optional PyTorch RNG generator for deterministic reproduction.

    Returns:
        x: (B, T) input token IDs
        y: (B, T) target token IDs, shifted by +1
    """
    max_idx = len(data) - context_length
    if max_idx <= 0:
        raise ValueError(f"Data length {len(data)} too short for context_length {context_length}")

    # Generate random starting indices
    indices = torch.randint(
        0,
        max_idx,
        (batch_size,),
        generator=generator,
    )

    x = torch.stack([data[i : i + context_length] for i in indices])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in indices])

    if device != "cpu":
        x = x.to(device)
        y = y.to(device)

    return x, y
