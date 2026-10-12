# Phase 5 Completion Report — Data Preparation, Training Engine & Milestone M4 Overfit Gate

> **Date**: 2026-10-13  
> **Phase**: 5 of 10  
> **Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 5 delivers the end-to-end data processing and training infrastructure required to train the GPT model from scratch:
1. **Tiny Shakespeare Dataset Pipeline**: Cryptographically verified raw corpus download, deterministic 90/10 train/val splits without leakage, and training-only vocabulary derivation.
2. **Deterministic Character Tokenizer**: Fully reversible character-level tokenizer with JSON serialization, `<unk>` handling, and tensor encoding.
3. **Sequence Construction**: Causal shifted targets ($x = \text{tokens}[t : t+T]$, $y = \text{tokens}[t+1 : t+T+1]$) with random batch sampling.
4. **Training Engine**: Modular training loop with AdamW optimizer, cosine learning rate schedule with linear warmup, gradient clipping, validation loss/perplexity evaluation, sample generation callback, and atomic checkpoints.
5. **Milestone M4 Overfit Gate**: Executed and verified deliberate overfit check on a fixed sequence, driving cross-entropy loss from **2.7518 down to 0.000308** with **100.0% next-token accuracy** in under 1 second.
6. **Testing & Audit**: 16 new unit tests added; full suite passes at **320/320 tests**; repository audit passes with 84 tracked files and 0 findings.

---

## 2. Dataset Provenance & Split Integrity

All dataset provenance is persisted in [`data/provenance.json`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/data/provenance.json):

* **Source URL**: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
* **Raw SHA256**: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
* **Corpus Size**: 1,115,394 bytes, 40,000 lines
* **Split Boundary**:
  * Training Split: 1,003,854 characters (90.0%)
  * Validation Split: 111,540 characters (10.0%)
* **Vocabulary**: Exactly 65 unique characters fitted strictly on training data:
  ```
  ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  ```
* **Unseen Characters in Validation**: 0 (empty list `[]`, confirming 100% vocabulary coverage without leakage).

---

## 3. Milestone M4: Tiny-Subset Overfit Gate Verification

Executed via [`experiments/tiny_overfit.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/experiments/tiny_overfit.py) and verified in [`tests/test_overfit.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/tests/test_overfit.py):

* **Test Sequence**: `"To be, or not to be, that is the question:"` (41 tokens context)
* **Model Configuration**: $d_{\text{model}}=64$, $n_{\text{layers}}=2$, $n_{\text{heads}}=2$, $d_{\text{ffn}}=128$, $\text{dropout}=0.0$
* **Optimizer**: AdamW ($\text{lr}=0.01$, $\text{weight\_decay}=0.0$)
* **Measured Progression**:
  * Step 0 (Untrained initial): **2.7518** (matches theoretical $\ln(V) = \ln(16) \approx 2.7726$)
  * Step 20: **0.246136**
  * Step 40: **0.003118**
  * Step 60: **0.000525**
  * Step 80: **0.000308**
* **Total Loss Reduction**: $2.7518 \to 0.000308$ ($\Delta = 2.7515$)
* **Next-Token Accuracy**: **100.0%** (41 / 41 tokens perfectly predicted)
* **Execution Time**: 0.86 seconds
* **Conclusion**: Proves that causal masking, next-token off-by-one shifted targets, linear projections, Pre-LN residual paths, autograd, and AdamW optimizer are mathematically sound and operating without bug or gradient blockage.

---

## 4. Deliverable File Inventory

| Component | File Path | Description | Status |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | [`src/zaid_lm/tokenizer.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/src/zaid_lm/tokenizer.py) | `CharacterTokenizer` class with fit, encode, decode, JSON save/load | ✅ Verified |
| **Dataset Loader** | [`src/zaid_lm/dataset.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/src/zaid_lm/dataset.py) | `ShakespeareDataset`, `download_tiny_shakespeare`, `prepare_dataset`, `get_batch` | ✅ Verified |
| **Checkpointing** | [`src/zaid_lm/checkpoint.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/src/zaid_lm/checkpoint.py) | Atomic `save_checkpoint` and `load_checkpoint` with state dicts, config, RNG | ✅ Verified |
| **Training Engine** | [`src/zaid_lm/train.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/src/zaid_lm/train.py) | Full `train()` function with cosine schedule, gradient clipping, validation loss/PPL | ✅ Verified |
| **Overfit Script** | [`experiments/tiny_overfit.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/experiments/tiny_overfit.py) | Reproducible M4 overfit verification script | ✅ Verified |
| **Data Provenance** | [`data/provenance.json`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/data/provenance.json) | Cryptographic corpus provenance record | ✅ Verified |
| **Unit Tests** | [`tests/test_tokenizer.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/tests/test_tokenizer.py) | 9 tests covering tokenizer roundtrip, bounds, `<unk>`, persistence | ✅ 9/9 Pass |
| **Unit Tests** | [`tests/test_dataset.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/tests/test_dataset.py) | 4 tests covering shifted target alignment, batching, shapes | ✅ 4/4 Pass |
| **Unit Tests** | [`tests/test_checkpoint.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/tests/test_checkpoint.py) | 2 tests verifying atomic saving and exact reload parity | ✅ 2/2 Pass |
| **Unit Tests** | [`tests/test_overfit.py`](file:///c:/Users/hp%20z%20book/Desktop/ai%20assignment%201/tests/test_overfit.py) | 1 test asserting M4 loss drop and 100% accuracy | ✅ 1/1 Pass |

---

## 5. Quality & Audit Gates

* **Pytest Suite**: **320 tests passed** in 15.70s (100% passing across all 12 test files).
* **Repository Audit (`scripts/verify_repository.py`)**: **PASS** (84 tracked files, 0 findings).
* **Ruff Linter & Formatter**: 100% compliant (`ruff check .` passes with 0 errors; 36 files properly formatted).

---

## 6. Next Phase: Phase 6 (Train Genuine Compact GPT Model)

According to master instructions, Phase 5 concludes with stopping and requesting approval before running full training:
* **Preconditions**:
  * Model architecture: 4 layers, 4 heads, $d_{\text{model}}=128$, context length 128 (809,984 parameters).
  * Training parameters: batch size 16 or 32, train steps 1,000–2,000, AdamW lr 3e-4 with cosine decay.
  * Deliverables in Phase 6: Figure F06 (real train vs val loss curves), Figure F07 (trained attention heatmaps), Figure F08 (5 authentic generated samples from fixed prompts), and persistent checkpoint in `checkpoints/best_model.pt`.
