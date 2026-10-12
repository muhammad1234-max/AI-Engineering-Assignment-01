"""Atomic model checkpoint saving and loading with full metadata provenance."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from zaid_lm.config import ProjectConfig
from zaid_lm.gpt import GPT
from zaid_lm.tokenizer import CharacterTokenizer


def save_checkpoint(
    checkpoint_path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: ProjectConfig,
    tokenizer: CharacterTokenizer | None = None,
    step: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save an atomic, complete checkpoint to disk.

    Writes to a temporary file in the same directory first, then renames atomically.
    Preserves model weights, optimizer state, config, tokenizer, step, losses, and RNG state.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config dataclass to serializable dict
    config_dict = asdict(config)
    # Ensure Path objects are converted to strings
    config_dict["output_path"] = str(config_dict["output_path"])

    checkpoint_data: dict[str, Any] = {
        "format_version": "1.0",
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config_dict,
        "model_state_dict": model.state_dict(),
        "rng_state": torch.get_rng_state(),
        "extra_metadata": extra_metadata or {},
    }

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

    if tokenizer is not None:
        checkpoint_data["tokenizer"] = {
            "vocabulary": tokenizer.vocabulary,
            "unk_token": tokenizer.unk_token,
            "vocab_size": tokenizer.vocab_size,
        }

    # Atomic write pattern
    temp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint from disk, optionally restoring weights into model and optimizer.

    If model is provided, model.load_state_dict is called and verified.
    If model is None, a fresh GPT model is constructed using the saved config.

    Returns:
        Dictionary containing loaded checkpoint metadata, model, optimizer, config, and tokenizer.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_config = checkpoint_data.get("config", {})
    if "output_path" in raw_config:
        raw_config["output_path"] = Path(raw_config["output_path"])
    config = ProjectConfig(**raw_config)

    # Reconstruct or restore model
    if model is None:
        model = GPT(config)
        model.to(device)

    model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)

    # Restore optimizer state if provided and present
    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Restore tokenizer if saved
    tokenizer = None
    if "tokenizer" in checkpoint_data:
        tok_data = checkpoint_data["tokenizer"]
        tokenizer = CharacterTokenizer(
            vocabulary=tok_data["vocabulary"],
            unk_token=tok_data.get("unk_token"),
        )

    # Restore RNG state if available
    if "rng_state" in checkpoint_data:
        try:
            torch.set_rng_state(checkpoint_data["rng_state"])
        except Exception:
            pass  # Device or version mismatch guard

    return {
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "tokenizer": tokenizer,
        "step": checkpoint_data.get("step", 0),
        "train_loss": checkpoint_data.get("train_loss"),
        "val_loss": checkpoint_data.get("val_loss"),
        "extra_metadata": checkpoint_data.get("extra_metadata", {}),
    }
