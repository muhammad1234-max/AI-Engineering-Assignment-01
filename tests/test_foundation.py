import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import zaid_lm
from zaid_lm.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def draw_random_values():
    return random.random(), np.random.random(5), torch.rand(5)


def test_seed_repeats_all_cpu_rngs():
    seed_everything(42)
    first = draw_random_values()
    seed_everything(42)
    second = draw_random_values()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    seed_everything(43)
    third = draw_random_values()
    assert first[0] != third[0]
    assert not torch.equal(first[2], third[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_cpu_seed_never_calls_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU smoke path must not touch CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    seed_everything(7, device="cpu")


def test_unavailable_cuda_has_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        seed_everything(42, device="cuda")


@pytest.mark.parametrize("seed", [-1, True, 2**32, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)


def test_seed_mode_controls():
    with pytest.raises(ValueError, match="device"):
        seed_everything(42, device="auto")
    with pytest.raises(ValueError, match="boolean"):
        seed_everything(42, deterministic=1)
    seed_everything(42, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(42)


def test_package_is_installed_and_importable_from_outside_repo(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import zaid_lm; print(zaid_lm.ProjectConfig().to_dict())"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "context_length" in result.stdout
    assert zaid_lm.__file__ is not None


def test_cli_parses_file_and_overrides_without_creating_outputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zaid_lm",
            "--config",
            str(ROOT / "configs/baseline.json"),
            "--device",
            "cpu",
            "--seed",
            "123",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "proposal_only"
    assert receipt["config"]["seed"] == 123
    assert receipt["config"]["device"] == "cpu"
    assert not list(tmp_path.iterdir())


def test_cli_rejects_invalid_override():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zaid_lm",
            "--config",
            str(ROOT / "configs/baseline.json"),
            "--seed",
            "-1",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "seed" in result.stderr


def test_structure_and_config_paths():
    for relative in (
        "configs/baseline.json",
        "src/zaid_lm/config.py",
        "tests",
        "experiments",
        "notebooks",
        "api",
        "docs/assignment_requirements.md",
        "docs/phase_log.md",
        "report",
        "data",
        "checkpoints",
        "outputs/figures",
        "outputs/logs",
        "outputs/samples",
    ):
        assert (ROOT / relative).exists(), relative
    config = zaid_lm.load_config(ROOT / "configs/baseline.json")
    assert (ROOT / config.output_path).is_dir()
