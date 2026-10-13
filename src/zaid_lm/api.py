"""FastAPI REST API for serving Shakespearean LLM causal language model inference."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from zaid_lm.checkpoint import load_checkpoint

# Global model state
_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "metadata": {},
    "device": "cpu",
}


def get_checkpoint_path() -> Path:
    """Resolve checkpoint file path from environment or default locations."""
    env_path = os.getenv("ZAID_LM_CHECKPOINT")
    if env_path:
        return Path(env_path)

    default_paths = [
        Path("checkpoints/best_model.pt"),
        Path("checkpoints/final_model.pt"),
    ]
    for p in default_paths:
        if p.exists():
            return p
    return default_paths[0]


def load_model_state():
    """Load model checkpoint into memory."""
    ckpt_path = get_checkpoint_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not ckpt_path.exists():
        download_url = os.getenv("ZAID_LM_CHECKPOINT_URL")
        if download_url:
            print(f"Downloading checkpoint from {download_url}...")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(download_url, ckpt_path)
            print(f"Downloaded checkpoint to {ckpt_path}.")

    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, device=device)
        _STATE["model"] = ckpt["model"]
        _STATE["tokenizer"] = ckpt["tokenizer"]
        _STATE["config"] = ckpt["config"]
        _STATE["metadata"] = {
            "checkpoint_name": ckpt_path.name,
            "step": ckpt["step"],
            "val_loss": ckpt["val_loss"],
            "train_loss": ckpt["train_loss"],
        }
        _STATE["device"] = device
        _STATE["model"].eval()
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_state()
    yield


# Initialize FastAPI application
app = FastAPI(
    title="Shakespearean LLM — Educational GPT Inference API",
    description="REST API serving from-scratch trained Pre-LN causal Transformer on Tiny Shakespeare.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Conditioning text prompt",
        json_schema_extra={"example": "ROMEO:\n"},
    )
    max_new_tokens: int = Field(80, ge=1, le=512, description="Maximum new tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(40, ge=0, le=100, description="Top-k filtering (0 disables)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling threshold")
    repetition_penalty: float = Field(1.05, ge=1.0, le=2.0, description="Repetition penalty")
    seed: int | None = Field(None, description="Optional RNG seed for reproducible generation")


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    model_step: int
    val_loss: float | None


class ModelInfoResponse(BaseModel):
    model_name: str
    parameters: int
    architecture: str
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    d_model: int
    ffn_dim: int
    device: str
    checkpoint: dict[str, Any]


@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Welcome to Shakespearean LLM Causal Language Model API",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/generate"],
    }


@app.get("/health", tags=["Info"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _STATE["model"] is not None,
        "device": _STATE["device"],
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    if _STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")

    cfg = _STATE["config"]
    return {
        "model_name": "Shakespearean LLM Tiny-Shakespeare GPT",
        "parameters": _STATE["model"].count_parameters(),
        "architecture": "Pre-LN Causal Decoder-Only Transformer",
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "ffn_dim": cfg.ffn_dim,
        "device": _STATE["device"],
        "checkpoint": _STATE["metadata"],
    }


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_completion(req: GenerateRequest):
    if _STATE["model"] is None or _STATE["tokenizer"] is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")

    model = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    if req.seed is not None:
        torch.manual_seed(req.seed)

    prompt = req.prompt.replace("\\n", "\n").replace("\\t", "\t")
    # Filter prompt characters to known vocabulary
    filtered = [ch for ch in prompt if ch in tokenizer.char2idx]
    if not filtered:
        filtered = ["\n"]
    clean_prompt = "".join(filtered)

    prompt_ids = tokenizer.encode(clean_prompt, as_tensor=True).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k if req.top_k > 0 else None,
            top_p=req.top_p if req.top_p < 1.0 else None,
            repetition_penalty=req.repetition_penalty,
        )
    elapsed = time.time() - start_time

    full_text = tokenizer.decode(out_ids[0].cpu())
    new_tokens = out_ids.size(1) - prompt_ids.size(1)
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "prompt": req.prompt,
        "completion": full_text[len(clean_prompt) :],
        "full_text": full_text,
        "tokens_generated": new_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tok_per_sec, 2),
        "model_step": _STATE["metadata"].get("step", 0),
        "val_loss": _STATE["metadata"].get("val_loss"),
    }
