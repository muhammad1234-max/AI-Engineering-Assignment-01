"""Unit tests verifying Milestone M4 tiny-subset overfit gate."""

from __future__ import annotations

from experiments.tiny_overfit import run_tiny_overfit


def test_tiny_overfit_gate():
    """Verify that a small GPT model overfits on a single fixed batch."""
    result = run_tiny_overfit(steps=70, lr=0.01, seed=42, verbose=False)
    assert result["status"] == "PASS"
    assert result["initial_loss"] > 2.0
    assert result["final_loss"] < 0.05
    assert result["accuracy"] == 100.0
    assert result["loss_reduction"] > 2.0
