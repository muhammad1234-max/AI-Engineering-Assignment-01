# Phase 6 Completion Report — Model Training & Evidence Generation

> **Date**: 2026-10-13  
> **Phase**: 6 of 10  
> **Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 6 executes the full-scale from-scratch training of our compact causal language model (`zaid_lm` GPT) on Tiny Shakespeare using a Tesla T4 GPU (Google Colab) and produces all required empirical evidence deliverables:
* **Model**: 4 decoder layers, 4 attention heads, $d_{\text{model}}=128$, $d_{\text{ffn}}=512$, context length 128 (809,984 parameters).
* **Optimization**: AdamW ($\text{lr}=3\times 10^{-4}$ with linear warmup and cosine decay to $3\times 10^{-5}$), batch size 32, gradient clipping norm 1.0.
* **Duration**: 1,500 steps completed in **40.58 seconds** on GPU (~150,000 tokens/second throughput).
* **Convergence**: Cross-entropy validation loss reduced from **3.357 to 2.3384** (Validation Perplexity: **10.36**).
* **Delivered Artifacts**:
  * Checkpoint: `checkpoints/best_model.pt` (verified locally on CPU for independent inference).
  * Evidence Figure **F06**: Loss & perplexity curves (`report/evidence/F06_loss.png`).
  * Evidence Figure **F07**: Trained attention heatmaps (`report/evidence/F07_trained_attention.png`).
  * Evidence Figure **F08**: Authentic generated samples from 5 fixed prompts (`report/evidence/F08_samples.png` and `outputs/samples/generated_samples.txt`).

---

## 2. Training Metrics & History

All step metrics are saved in `outputs/training_history.json`:

| Step | Train Loss | Validation Loss | Validation Perplexity | Learning Rate | Elapsed Time | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 3.2455 | 3.3570 | 28.70 | $2.99 \times 10^{-4}$ | 3.37s | 121,471 tok/s |
| 300 | 2.7901 | 2.7224 | 15.22 | $2.81 \times 10^{-4}$ | 8.45s | 145,360 tok/s |
| 600 | 2.5714 | 2.5021 | 12.21 | $2.26 \times 10^{-4}$ | 16.03s | 153,280 tok/s |
| 900 | 2.4532 | 2.4180 | 11.22 | $1.52 \times 10^{-4}$ | 24.11s | 152,900 tok/s |
| 1200 | 2.3810 | 2.3690 | 10.69 | $7.80 \times 10^{-5}$ | 32.05s | 153,400 tok/s |
| **1500** | **2.3746** | **2.3384** | **10.36** | **$3.00 \times 10^{-5}$** | **40.23s** | **152,720 tok/s** |

---

## 3. Evidence Artifacts

1. **Figure F06 (`report/evidence/F06_loss.png`)**:
   * Shows smooth, monotonic decrease in cross-entropy loss and perplexity.
   * Annotated with best validation step (Step 1,500, loss 2.338).
   * Verified by SHA256 receipt in `report/evidence/F06_loss.json`.
2. **Figure F07 (`report/evidence/F07_trained_attention.png`)**:
   * Extracted softmax attention weights from Layer 0 (early) and Layer 3 (late) across all 4 heads for prompt `"ROMEO:\nWhat light"`.
   * Proves causal masking: strictly zero probabilities in the upper triangle.
   * Demonstrates specialized attention patterns (recency bias, character repetition, punctuation tracking).
3. **Figure F08 (`report/evidence/F08_samples.png`)**:
   * 5 fixed prompts generated with temperature 0.8, top-k 40:
     * Prompt 1: `"ROMEO:\n"`
     * Prompt 2: `"JULIET:\n"`
     * Prompt 3: `"HAMLET:\n"`
     * Prompt 4: `"To be, or not to be"`
     * Prompt 5: `"O, swear not by the moon"`
   * Shows clear emergence of theatrical dialogue structures, character capitalization, and archaic English rhythm without external post-processing.
