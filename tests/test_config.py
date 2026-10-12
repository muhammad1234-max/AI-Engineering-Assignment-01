import json
from dataclasses import replace
from pathlib import Path

import pytest

from zaid_lm.config import ProjectConfig, load_config


def test_file_roundtrip_and_path_conversion(tmp_path):
    config = ProjectConfig(vocab_size=7, output_path=tmp_path / "results")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path) == config
    assert isinstance(load_config(path).output_path, Path)


def test_baseline_loads_with_unresolved_vocabulary():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/baseline.json")
    with pytest.raises(ValueError, match="unresolved"):
        config.validate_model_inputs(vocab_size=9, sequence_length=3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 0),
        ("vocab_size", True),
        ("context_length", -1),
        ("context_length", 1.5),
        ("d_model", 0),
        ("n_heads", 0),
        ("n_layers", -3),
        ("ffn_dim", False),
        ("batch_size", "16"),
        ("train_steps", 0),
        ("max_new_tokens", -1),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", True),
        ("dropout", -0.01),
        ("dropout", 1),
        ("dropout", float("nan")),
        ("dropout", True),
        ("temperature", 0),
        ("temperature", -1),
        ("temperature", float("inf")),
        ("temperature", "0.8"),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("optimizer", "unknown"),
        ("optimizer", []),
        ("device", "auto"),
        ("tokenizer", "pretrained"),
        ("activation", "unknown"),
        ("position_encoding", "learned"),
        ("output_path", " "),
        ("output_path", 9),
    ],
)
def test_invalid_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        replace(ProjectConfig(), **{field: value})


def test_divisibility_is_enforced():
    with pytest.raises(ValueError, match="divisible"):
        ProjectConfig(d_model=130, n_heads=4)
    assert ProjectConfig(d_model=144, n_heads=6).d_model // 6 == 24


def test_valid_numeric_boundaries():
    ProjectConfig(dropout=0, temperature=0.01, seed=0)
    ProjectConfig(dropout=0.99, temperature=3, seed=2**32 - 1)


def test_model_input_compatibility():
    config = ProjectConfig(vocab_size=11, context_length=8)
    config.validate_model_inputs(vocab_size=11, sequence_length=8)
    with pytest.raises(ValueError, match="vocabulary"):
        config.validate_model_inputs(vocab_size=12, sequence_length=8)
    with pytest.raises(ValueError, match="context_length"):
        config.validate_model_inputs(vocab_size=11, sequence_length=9)
    with pytest.raises(ValueError, match="sequence_length"):
        config.validate_model_inputs(vocab_size=11, sequence_length=0)


@pytest.mark.parametrize(
    "content",
    [
        '{"typo": 3}',
        '{"n_heads": 4, "n_heads": 8}',
        "[]",
        "{bad",
    ],
)
def test_malformed_or_ambiguous_file_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")
