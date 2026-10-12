"""Strict configuration for the future model; no model or training is executed."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    # None means unresolved until a tokenizer is fitted on training data.
    vocab_size: int | None = None
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    optimizer: str = "adamw"
    learning_rate: float = 0.0003
    batch_size: int = 16
    train_steps: int = 1000
    temperature: float = 0.8
    max_new_tokens: int = 80
    device: str = "cpu"
    seed: int = 42
    output_path: Path = Path("outputs")
    tokenizer: str = "character"
    activation: str = "gelu"
    position_encoding: str = "sinusoidal"

    def __post_init__(self) -> None:
        for name in (
            "context_length",
            "d_model",
            "n_heads",
            "n_layers",
            "ffn_dim",
            "batch_size",
            "train_steps",
            "max_new_tokens",
        ):
            self._positive_int(name, getattr(self, name))
        if self.vocab_size is not None:
            self._positive_int("vocab_size", self.vocab_size)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("dropout", "learning_rate", "temperature"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive; greedy decoding is a separate mode")
        for name, choices in {
            "optimizer": {"adamw", "adam", "sgd"},
            "device": {"cpu", "cuda", "mps"},
            "tokenizer": {"character"},
            "activation": {"gelu", "relu"},
            "position_encoding": {"sinusoidal"},
        }.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if not isinstance(self.output_path, (str, Path)) or not str(self.output_path).strip():
            raise ValueError("output_path must be a nonempty string or Path")
        object.__setattr__(self, "output_path", Path(self.output_path))

    @staticmethod
    def _positive_int(name: str, value: Any) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ProjectConfig":
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_path"] = str(self.output_path)
        return values

    def validate_model_inputs(self, *, vocab_size: int, sequence_length: int) -> None:
        """Guard future tokenizer/model compatibility without implementing either."""
        self._positive_int("vocab_size", vocab_size)
        self._positive_int("sequence_length", sequence_length)
        if self.vocab_size is None:
            raise ValueError("vocab_size is unresolved; fit the training tokenizer first")
        if self.vocab_size != vocab_size:
            raise ValueError("configured vocab_size does not match tokenizer vocabulary")
        if sequence_length > self.context_length:
            raise ValueError("sequence_length exceeds context_length")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def load_config(path: str | Path) -> ProjectConfig:
    """Load UTF-8 JSON, rejecting duplicate fields and invalid values."""
    with Path(path).open(encoding="utf-8") as stream:
        values = json.load(stream, object_pairs_hook=_unique_object)
    return ProjectConfig.from_dict(values)
