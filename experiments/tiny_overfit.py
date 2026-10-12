"""Milestone M4: Tiny-subset overfit gate to verify training engine mechanics.

This script proves that:
1. Shifted targets x[t] -> y[t] = x[t+1] are correctly aligned.
2. Causal masking does not obstruct information from past tokens.
3. Gradient backpropagation flows through all embeddings, attention, FFN, and LM head.
4. AdamW optimizer successfully drives cross-entropy loss down toward zero on a tiny fixed batch.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from zaid_lm.config import ProjectConfig
from zaid_lm.gpt import GPT
from zaid_lm.tokenizer import CharacterTokenizer


def run_tiny_overfit(
    *,
    steps: int = 80,
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run overfit verification on a fixed tiny string."""
    torch.manual_seed(seed)

    # Fixed training text (short, realistic snippet)
    sample_text = "To be, or not to be, that is the question:"
    tokenizer = CharacterTokenizer.from_text(sample_text)

    # Encode sequence
    tokens = tokenizer.encode(sample_text, as_tensor=True)
    context_length = len(tokens) - 1

    # Single fixed batch: x is tokens[:-1], y is tokens[1:]
    x = tokens[:context_length].unsqueeze(0)  # (1, T)
    y = tokens[1 : context_length + 1].unsqueeze(0)  # (1, T)

    # Small GPT model configured for this overfit test
    config = ProjectConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        d_model=64,
        n_heads=2,
        n_layers=2,
        ffn_dim=128,
        dropout=0.0,  # No dropout for deterministic overfit
        learning_rate=lr,
        device="cpu",
    )

    model = GPT(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # Initial loss
    with torch.no_grad():
        initial_logits, initial_loss = model(x, targets=y)
    initial_loss_val = initial_loss.item()

    if verbose:
        print("=" * 65)
        print("Milestone M4 — Tiny Subset Overfit Gate")
        print("=" * 65)
        print(f"Sample text: {repr(sample_text)}")
        print(f"Vocab size:  {tokenizer.vocab_size} unique characters")
        print(f"Context len: {context_length} tokens")
        print(
            f"Model:       d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}"
        )
        print(
            f"Initial loss (untrained): {initial_loss_val:.4f} (theoretical ln(V) = {math.log(tokenizer.vocab_size):.4f})"
        )
        print("-" * 65)

    loss_history: list[float] = [initial_loss_val]
    start_time = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and (step % 20 == 0 or step == steps):
            print(f"Step {step:3d}/{steps} | Loss: {loss_val:.6f}")

    elapsed = time.time() - start_time
    final_loss_val = loss_history[-1]

    # Test reconstruction on the trained model
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        predicted_ids = final_logits.argmax(dim=-1).squeeze(0)
        # Shifted targets comparison
        reconstructed_chars = [tokenizer.idx2char[int(idx)] for idx in predicted_ids]
        target_chars = [tokenizer.idx2char[int(idx)] for idx in y.squeeze(0)]
        matches = sum(p == t for p, t in zip(reconstructed_chars, target_chars, strict=True))
        accuracy = matches / len(target_chars) * 100.0

    if verbose:
        print("-" * 65)
        print(f"Final loss after {steps} steps: {final_loss_val:.6f}")
        print(
            f"Loss reduction: {initial_loss_val:.4f} -> {final_loss_val:.4f} (drop = {initial_loss_val - final_loss_val:.4f})"
        )
        print(f"Target characters:    {''.join(target_chars)}")
        print(f"Predicted characters: {''.join(reconstructed_chars)}")
        print(f"Next-token accuracy:  {accuracy:.1f}% ({matches}/{len(target_chars)})")
        print(f"Execution time:       {elapsed:.2f}s")
        print("=" * 65)

    assert final_loss_val < 0.1, (
        f"Overfit gate failed! Final loss {final_loss_val:.4f} did not drop below 0.1"
    )

    return {
        "status": "PASS",
        "initial_loss": initial_loss_val,
        "final_loss": final_loss_val,
        "loss_reduction": initial_loss_val - final_loss_val,
        "accuracy": accuracy,
        "history": loss_history,
        "steps": steps,
        "time_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone M4 Tiny-Subset Overfit Gate")
    parser.add_argument("--steps", type=int, default=80, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()
    run_tiny_overfit(steps=args.steps, lr=args.lr)


if __name__ == "__main__":
    main()
