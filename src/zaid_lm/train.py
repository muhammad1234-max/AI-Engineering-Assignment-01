"""Training engine for GPT language model on Tiny Shakespeare."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from zaid_lm.checkpoint import save_checkpoint
from zaid_lm.config import ProjectConfig, load_config
from zaid_lm.dataset import get_batch, prepare_dataset
from zaid_lm.gpt import GPT
from zaid_lm.tokenizer import CharacterTokenizer


def get_lr(step: int, total_steps: int, lr: float, warmup_steps: int = 50) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return lr * 0.1
    # Cosine decay between lr and 0.1 * lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = lr * 0.1
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_iters: int = 20,
    device: str = "cpu",
) -> float:
    """Compute average cross-entropy loss over multiple evaluation batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device=device)
        _, loss = model(x, targets=y)
        losses[k] = loss.item()
    model.train()
    return float(losses.mean().item())


def train(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ProjectConfig,
    tokenizer: CharacterTokenizer,
    *,
    eval_interval: int = 100,
    eval_iters: int = 20,
    checkpoint_dir: Path | str = "checkpoints",
    sample_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train GPT causal language model with AdamW, validation, and checkpointing.

    Args:
        model: GPT model instance.
        train_data: 1D tensor of training token IDs.
        val_data: 1D tensor of validation token IDs.
        config: ProjectConfig specifying hyperparameters.
        tokenizer: CharacterTokenizer for decoding sample generations.
        eval_interval: Number of steps between validation evaluations.
        eval_iters: Batches to average during each evaluation.
        checkpoint_dir: Directory for storing model checkpoints.
        sample_callback: Optional callback receiving generated text samples.
        log_callback: Optional callback receiving step metrics dictionary.

    Returns:
        Dictionary containing training history, final losses, runtime, and parameter counts.
    """
    device = config.device
    model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure AdamW with weight decay
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    start_time = time.time()
    total_tokens_seen = 0

    print(
        f"Starting training on {device} | Total steps: {config.train_steps} | Model params: {model.count_parameters():,}"
    )

    for step in range(1, config.train_steps + 1):
        # Update learning rate with schedule
        current_lr = get_lr(step, config.train_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Fetch training batch
        x, y = get_batch(train_data, config.batch_size, config.context_length, device=device)
        total_tokens_seen += config.batch_size * config.context_length

        # Forward pass and loss
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic evaluation
        if step % eval_interval == 0 or step == config.train_steps:
            val_loss = estimate_loss(
                model,
                val_data,
                config.batch_size,
                config.context_length,
                eval_iters=eval_iters,
                device=device,
            )
            perplexity = math.exp(min(val_loss, 20.0))  # guard overflow
            elapsed = time.time() - start_time
            tps = total_tokens_seen / elapsed if elapsed > 0 else 0.0

            log_entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "val_perplexity": round(perplexity, 4),
                "learning_rate": round(current_lr, 6),
                "elapsed_seconds": round(elapsed, 2),
                "tokens_per_second": round(tps, 1),
            }
            history.append(log_entry)

            print(
                f"Step {step:4d}/{config.train_steps} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {perplexity:.2f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

            if log_callback is not None:
                log_callback(log_entry)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=tokenizer,
                    step=step,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    extra_metadata={"best_val_loss": best_val_loss, "perplexity": perplexity},
                )

            # Periodic sample generation hook
            if sample_callback is not None and (
                step % (eval_interval * 2) == 0 or step == config.train_steps
            ):
                prompt = "KING:"
                prompt_ids = tokenizer.encode(prompt, as_tensor=True).unsqueeze(0).to(device)
                model.eval()
                gen_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=40,
                    temperature=0.8,
                    top_k=20,
                )
                sample_text = tokenizer.decode(gen_ids[0].cpu())
                sample_callback(sample_text)

    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir / "final_model.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=config.train_steps,
        train_loss=history[-1]["train_loss"] if history else None,
        val_loss=history[-1]["val_loss"] if history else None,
    )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "total_time_seconds": time.time() - start_time,
        "total_tokens_seen": total_tokens_seen,
        "parameters": model.count_parameters(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train GPT Language Model on Tiny Shakespeare")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline.json"), help="Config file path"
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda)")
    parser.add_argument("--steps", type=int, default=None, help="Train steps override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--eval-interval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Outputs directory"
    )
    args = parser.parse_args()

    # Load and override config
    config = load_config(args.config)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    elif torch.cuda.is_available():
        overrides["device"] = "cuda"
    else:
        overrides["device"] = "cpu"

    if args.steps is not None:
        overrides["train_steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    # Prepare dataset
    print(f"Preparing dataset from {args.data_dir}...")
    splits = prepare_dataset(args.data_dir)
    overrides["vocab_size"] = splits.tokenizer.vocab_size
    config = replace(config, **overrides)

    print(
        f"Initializing GPT model (vocab_size={config.vocab_size}, d_model={config.d_model}, layers={config.n_layers})..."
    )
    torch.manual_seed(config.seed)
    model = GPT(config)

    # Output directory setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_file = args.output_dir / "training_history.json"

    def sample_logger(text: str):
        print(f"\n--- Generated Sample ---\n{text}\n------------------------\n", flush=True)

    result = train(
        model=model,
        train_data=splits.train_tokens,
        val_data=splits.val_tokens,
        config=config,
        tokenizer=splits.tokenizer,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_callback=sample_logger,
    )

    history_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nTraining completed! History saved to {history_file}")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Total training time:  {result['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
