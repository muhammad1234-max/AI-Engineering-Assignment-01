"""Unit tests for the command-line inference engine."""

from __future__ import annotations

from pathlib import Path

from zaid_lm.generate import generate_text


def test_generate_text_function():
    ckpt_path = Path("checkpoints/best_model.pt")
    assert ckpt_path.exists(), "Trained model checkpoint must exist"

    prompt = "KING:"
    res = generate_text(
        ckpt_path,
        prompt,
        max_new_tokens=25,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        seed=42,
    )

    assert res["prompt"] == prompt
    assert len(res["completion"]) > 0
    assert res["new_tokens"] == 25
    assert res["tokens_per_second"] > 0.0
    assert res["full_text"].startswith(prompt)


def test_generate_text_unescape_and_clean():
    ckpt_path = Path("checkpoints/best_model.pt")
    # Prompt with literal escaped newline and character outside vocabulary
    prompt = "HAMLET:\\n@#"
    res = generate_text(ckpt_path, prompt, max_new_tokens=15, seed=42)

    assert len(res["full_text"]) > 0
    assert res["new_tokens"] == 15
