# Shakespearean LLM — From-Scratch Causal Transformer

[![CI](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/workflows/ci.yml/badge.svg)](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61dafb.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/pytest-331%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An educational, understandable, and production-ready **Pre-LN Causal Decoder-Only Transformer** (809,984 parameters) built completely from mathematical primitives and trained from scratch on the Tiny Shakespeare corpus.

- **Author / Owner**: Zaid Bin Naveed (`zaidbinnaveed`)
- **Repository**: [https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01)
- **Course**: AI Engineering Assignment #01 — Transformer Deep Dive (`A1_AIE.pdf`)
- **Strict Academic Constraint**: Zero pretrained model weights, zero black-box high-level abstractions (`transformers` prohibited and verified absent).

---

## Executive Summary & Key Results

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | **Pre-LN Causal Transformer** | 4 layers, 4 heads, $d_{\text{model}}=128$, FFN dimension 512 |
| **Total Parameters** | **809,984** | Includes token embedding, 4 decoder blocks, LayerNorm, and LM head |
| **Context Length** | **128 characters** | Fixed sinusoidal positional encodings |
| **Vocabulary** | **65 characters** | Deterministic character-level tokenizer fitted on Tiny Shakespeare |
| **Final Validation Loss** | **2.3384** | Evaluated on 10% held-out test split |
| **Validation Perplexity** | **10.36** | $e^{2.3384}$, strictly measured on unseen validation tokens |
| **Generation Throughput** | **~105–125 tokens/sec** | CPU inference with temperature, top-$k$, top-$p$, and repetition penalty |
| **Test Suite Coverage** | **331 passed, 0 failed** | Unit, integration, gradient, shape, and causality verification |
| **Audit Gate** | **PASS (0 findings)** | Automated cryptographic and provenance audit via `verify_repository.py` |

---

## Architectural Highlights

1. **Numerical Scaled Dot-Product Attention** (`src/zaid_lm/attention.py`):
   - Fully independent scaled dot-product attention: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$.
   - Strictly enforced upper-triangular future-token mask ($M_{i,j} = -\infty$ for $j > i$).
   - Accompanied by a 7-step handwritten and typed derivation with exact numerical receipts ([`docs/manual_attention.md`](docs/manual_attention.md)).

2. **Explicit Multi-Head Attention** (`src/zaid_lm/multihead_attention.py`):
   - Configurable attention heads ($h=4$, $d_k=32$), explicit projection splitting, per-head scaled attention, concatenation, and output projection ($W_O$).

3. **Sinusoidal Positional Encodings** (`src/zaid_lm/positional_encoding.py`):
   - Fixed, deterministic sinusoidal encodings computed as persistent buffers ($PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$).

4. **Pre-LayerNorm Decoder Blocks** (`src/zaid_lm/transformer_block.py`):
   - Modern Pre-LN formulation for stable gradient propagation:
     $$x^{(1)} = x + \text{MHA}(\text{LayerNorm}(x))$$
     $$x^{(2)} = x^{(1)} + \text{FFN}(\text{LayerNorm}(x^{(1)}))$$
   - Feed-Forward Network with GELU nonlinearity expanding $128 \to 512 \to 128$.

5. **Autoregressive Sampling Engine** (`src/zaid_lm/generate.py`):
   - Supports temperature scaling ($T$), top-$k$ filtering, top-$p$ nucleus sampling, repetition penalty, and RNG seed reproducibility.

---

## Repository Guide

```
AI-Engineering-Assignment-01/
├── src/zaid_lm/              # Core causal Transformer library (zero external LM deps)
│   ├── attention.py          # Scaled dot-product attention & causal masking
│   ├── multihead_attention.py# 4-head attention projection & merge
│   ├── positional_encoding.py# Fixed sinusoidal positional encodings
│   ├── transformer_block.py  # Pre-LN decoder block with GELU FFN & residuals
│   ├── gpt.py                # Complete causal GPT model assembly
│   ├── tokenizer.py          # Deterministic character-level tokenizer (vocab=65)
│   ├── dataset.py            # Contiguous shifted sequence loader (x, y)
│   ├── train.py              # Full training loop, AdamW optimizer, LR scheduler
│   ├── checkpoint.py         # Checkpoint saving, reloading & state validation
│   ├── generate.py           # CLI text generation with sampling controls
│   └── api.py                # Production FastAPI REST service
├── configs/                  # Model configurations (configs/baseline.json)
├── frontend/                 # Interactive React / TanStack Start First Folio Studio
│   ├── src/components/       # UI components, ChatApp, Brand splash, telemetry HUD
│   ├── src/styles.css        # Theatrical obsidian/gold design system with tabular nums
│   └── package.json          # Vite + React 19 frontend package definition
├── experiments/              # Verification scripts, ablation studies & visualizers
├── report/evidence/          # All 10 required academic evidence figures F01-F10 + receipts
├── notebooks/                # Google Colab interactive notebook (zaid_lm_colab.ipynb)
├── tests/                    # Comprehensive unit & integration test suite (331 tests)
├── docs/                     # Academic compliance matrix, phase reports & mathematical derivations
├── checkpoints/              # Checkpoint directory (checkpoints/best_model.pt)
├── Dockerfile                # Lightweight container image for cloud deployment
├── render.yaml               # Infrastructure-as-code deployment specification
├── requirements.txt          # Pinned runtime dependencies
├── pyproject.toml            # Python packaging metadata & console entrypoints
└── scripts/                  # Automated verification and provenance auditor
```

---

## Academic Evidence Figures (F01–F10)

All 10 required evidence figures are verified and tracked with cryptographically signed JSON receipts in [`report/evidence/`](report/evidence/):

| Figure | Required Proof | Evidence Artifact | Status |
| :--- | :--- | :--- | :--- |
| **F01** | Hand-calculated attention worked example (7 steps) | [`report/evidence/F01_manual_attention.png`](report/evidence/F01_manual_attention.png) | **VERIFIED** |
| **F02** | Causal mask blocks future tokens (3 & 8 tokens) | [`report/evidence/F02_causal_mask.png`](report/evidence/F02_causal_mask.png) | **VERIFIED** |
| **F03** | Multi-head shape printout & causality trace | [`report/evidence/F03_multihead_shape_trace.txt`](report/evidence/F03_multihead_shape_trace.txt) | **VERIFIED** |
| **F04** | Sinusoidal positional encoding representations | [`report/evidence/F04_positional_encoding.png`](report/evidence/F04_positional_encoding.png) | **VERIFIED** |
| **F05** | Model summary & parameter breakdown (809,984 params) | [`report/evidence/F05_model_summary.png`](report/evidence/F05_model_summary.png) | **VERIFIED** |
| **F06** | Real training loss vs validation loss curves | [`report/evidence/F06_loss.png`](report/evidence/F06_loss.png) | **VERIFIED** |
| **F07** | Attention weight inspection from trained checkpoint | [`report/evidence/F07_trained_attention.png`](report/evidence/F07_trained_attention.png) | **VERIFIED** |
| **F08** | Five genuine autoregressive generation samples | [`report/evidence/F08_samples.png`](report/evidence/F08_samples.png) | **VERIFIED** |
| **F09** | Controlled ablation studies (Pre-LN, Residuals, Heads) | [`report/evidence/F09_ablations.png`](report/evidence/F09_ablations.png) | **VERIFIED** |
| **F10** | Live CLI and FastAPI inference demonstration | [`report/evidence/F10_inference.png`](report/evidence/F10_inference.png) | **VERIFIED** |

---

## Controlled Ablation Studies

Three controlled ablations were conducted to isolate the empirical impact of architectural decisions:

| Configuration | Parameters | Val Loss | Val Perplexity | Observed Convergence Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Pre-LN, 4 Heads, Res)** | **809,984** | **2.3384** | **10.36** | **Fastest, highly stable convergence** |
| **Ablation 1: Post-LN (No Pre-LN)** | 809,984 | 2.5120 | 12.33 | Slower early progress, higher final loss |
| **Ablation 2: No Residuals** | 809,984 | 3.1492 | 23.32 | Gradient degradation; failed to capture complex verse |
| **Ablation 3: Single-Head ($h=1$)** | 809,984 | 2.4418 | 11.49 | Reduced representational capacity across heads |

---

## Installation & Quickstart

### 1. Python Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run Inference via CLI

```bash
# Generate verse conditioned on Hamlet's prompt
python -m zaid_lm.generate --prompt "HAMLET:\nTo be, or not to be" --max-new-tokens 120 --temperature 0.8
```

### 3. Launch FastAPI Backend

```bash
# Start FastAPI inference service on port 8000
python -m zaid_lm.api
```

Visit the interactive Swagger UI at **`http://localhost:8000/docs`**.

### 4. Launch Interactive Web Studio (Frontend)

```bash
cd frontend
npm install
npm run dev
```

Open **`http://localhost:8080`** to interact with the First Folio Studio.

---

## Quality Assurance & Verification

```bash
# Run complete test suite (331 tests)
python -m pytest -q

# Check linting and formatting
python -m ruff check .
python -m ruff format --check .

# Run repository integrity and provenance audit
python scripts/verify_repository.py
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
