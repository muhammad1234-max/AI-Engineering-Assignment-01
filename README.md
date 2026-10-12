# Zaid's AI Engineering Assignment #01

Build a small LLM from scratch: Transformer Deep Dive.
Owner: **zaidbinnaveed**. Authorized repository:
https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.

The objective is an independently implemented, understandable decoder Transformer
trained on Tiny Shakespeare, with numerical attention derivations, controlled
experiments, and usable inference. The academic specification is `A1_AIE.pdf`
(11 pages); its source identity is recorded in [requirements](docs/assignment_requirements.md).

**Current status: Phase 3 multi-head attention and sinusoidal positions implemented.**
Configuration validation, CPU seed controls, configuration inspection, explicit
scaled dot-product attention and causal masking work. Independent scalar checks,
gradient tests and F01/F02 evidence accompany the complete typed derivation.
Independent learned Q/K/V/output projections, explicit head split/merge and fixed
sinusoidal position buffers now have scalar-reference tests and F03/F04 evidence.
Zaid's personal mathematical review remains required before submission (H07/M01).
The complete model, tokenizer, training, text generation and API remain future work.
The configuration CLI below does not generate text or train a model.

## Installation

Supported Python: 3.11 through 3.14, standard CPython. Local verification uses
Windows x64 and Python 3.14.0. CI checks Linux 3.11 and 3.14 separately; see the phase
log for actual outcomes. Use a fresh virtual environment, without global installs.

From the repository root, PowerShell:

```powershell
python -m venv .venv-phase1
.\.venv-phase1\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 0
.\.venv-phase1\Scripts\python.exe -m pip install -r requirements.txt --timeout 30 --retries 0
.\.venv-phase1\Scripts\python.exe -m pip install --no-build-isolation -e .
```

Linux/macOS equivalents (local PowerShell and actual CI outcomes recorded separately):

```bash
python3 -m venv .venv-phase1
.venv-phase1/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 0
.venv-phase1/bin/python -m pip install -r requirements.txt --timeout 30 --retries 0
.venv-phase1/bin/python -m pip install --no-build-isolation -e .
```

CPU-only PyTorch is intentional for these development phases. CUDA dependencies and hardware/run
qualification belong to an explicitly approved future phase. If a platform has no
compatible wheel, report the install error rather than running an unapproved GPU job.
Direct dependencies are pinned in `requirements.txt` and `pyproject.toml`; the
Phase 1 Windows environment is recorded in `docs/environment-windows-py314.txt`;
Phase 2 adds pinned Matplotlib 3.10.8, with its environment recorded separately.

## Development commands

```powershell
.\.venv-phase1\Scripts\python.exe -m pip check
.\.venv-phase1\Scripts\python.exe -m pytest -q
.\.venv-phase1\Scripts\python.exe -m ruff check .
.\.venv-phase1\Scripts\python.exe -m ruff format --check .
.\.venv-phase1\Scripts\python.exe -m zaid_lm --config configs/baseline.json --device cpu
.\.venv-phase1\Scripts\zaid-lm-config.exe --config configs/baseline.json --seed 123
.\.venv-phase1\Scripts\python.exe -m build --no-isolation
.\.venv-phase1\Scripts\python.exe scripts/verify_repository.py
```

The configuration command prints JSON with `status: proposal_only` and the validated
settings. It reads files and prints settings without creating model/data/output artifacts.
Use `--help` for its options. Configuration paths resolve from the current working
directory; `output_path` is a `pathlib.Path` and does not create directories on load.
Formatting for contributors: `python -m ruff format .`; lint: `python -m ruff check .`.
The GitHub workflow runs CPU checks only, with read-only repository permissions.

## Attention mathematics and evidence

From the repository root after installation:

```powershell
.\.venv-phase1\Scripts\python.exe -m pytest tests/test_attention.py -q
.\.venv-phase1\Scripts\python.exe -m experiments.attention_example --figure
.\.venv-phase1\Scripts\python.exe -m experiments.visualize_causal_mask
```

Linux equivalents use `.venv-phase1/bin/python`. The first experiment fails loudly
if any of seven manual/scalar/engine intermediates disagree, then generates F01.
The second verifies index-based causal visibility at 3 and 8 tokens, then generates
F02. Figures are written to ignored `outputs/figures/`; only the two small validated
PNGs and hash receipts are copied to tracked `report/evidence/`. Both are educational
evidence, without a trained model. Reproduction can vary pixels across fonts/platforms;
numeric comparisons and source/content hashes identify the evidence actually generated.

Read the [full arithmetic](docs/manual_attention.md) and
[reasoned RNN/Transformer comparison](docs/why_transformers.md).
The former includes every dot product, exponent/denominator, probability and weighted
output, both unrestricted and causal. Zaid must personally review and reproduce it.

```python
from zaid_lm.attention import causal_forbidden_mask, scaled_dot_product_attention

output, weights = scaled_dot_product_attention(q, k, v, causal=True)
```

Inputs are Q `(..., Tq, dk)`, K `(..., Tk, dk)`, V `(..., Tk, dv)`;
identical batch/head leading axes, float32/float64, finite values, same device.
Output is `(..., Tq, dv)`, weights `(..., Tq, Tk)`. Unmasked lengths may differ;
causal lengths must match with aligned indices. Optional `mask=` broadcasts to weights:
bool True blocks, or additive same-dtype 0/-inf. Masks combine with causal restrictions.
Empty/mismatched inputs, incompatible masks, all-blocked rows, nonfinite inputs and
overflowing scores fail explicitly. No learned projections, fused attention, dropout,
precision conversion or device fallback. Verification is CPU only.

## Phase 3 modules and evidence

The installation above already includes the pinned plotting and test dependencies.
From the repository root:

```powershell
.\.venv-phase1\Scripts\python.exe -m experiments.multihead_shape_trace
.\.venv-phase1\Scripts\python.exe -m experiments.visualize_positions
.\.venv-phase1\Scripts\python.exe -m pytest tests/test_multihead_attention.py tests/test_positional_encoding.py tests/test_phase3_integration.py -q
.\.venv-phase1\Scripts\python.exe -m pytest -q
```

Linux equivalents use `.venv-phase1/bin/python`. The trace asserts real projection,
split/permutation, per-head and merge shapes, parameter count, independent scalar
numerics, causality and gradients. F04 checks all baseline values against scalar
sin/cos before plotting. Inspect [F03](report/evidence/F03_multihead_shape_trace.txt)
and [F04](report/evidence/F04_positional_encoding.png), with neighboring JSON hash
receipts. Raw copies live in ignored `outputs/logs/` and `outputs/figures/`.
Generated images may vary by platform fonts; receipts identify the exact stored bytes.

```python
from zaid_lm.multihead_attention import MultiHeadSelfAttention
from zaid_lm.positional_encoding import SinusoidalPositionalEncoding

positions = SinusoidalPositionalEncoding(128, 128)
attention = MultiHeadSelfAttention(128, 4)
y, weights = attention(positions(x), return_attention=True)  # x: (B,T,128)
```

Both modules require matching input/module float32 or float64 dtype and device.
Position offsets are validated; causal attention defaults to True. Output dropout
and position-addition dropout default to zero and disable in eval. Read the precise
[multi-head contract](docs/multihead_attention.md) and
[positional contract](docs/positional_encoding.md). This is a demonstration composition;
the decoder block, full GPT, real training and deployed inference remain pending.
The [Phase 3 report](docs/phase3_completion_report.md) records verification and review.

## Configuration proposal

See [baseline rationale](docs/architecture.md). `configs/baseline.json` proposes:
character tokenizer, context 128, width 128, 4 heads, 4 blocks, FFN 512, GELU,
sinusoidal positions, dropout 0.1, AdamW at 0.0003, batch 16, temperature 0.8,
seed 42, CPU, and 1,000 provisional future updates. **These are not run results or
training authorization.** Vocabulary is `null` until fitted on training data.
Model compatibility checks reject unresolved/mismatched vocabulary and excessive
context. Hardware qualification must finalize batch size, steps, device and budget.

## Directory guide and planned architecture

| Location | Purpose and status |
| --- | --- |
| `src/zaid_lm/` | Installed configuration, seeds, inspection CLI, attention/masks, multi-head and sinusoidal positions |
| `configs/` | Validated baseline proposal |
| `tests/` | Preserved 136 Phase 1/2 tests plus multi-head, position and composition tests |
| `docs/` | Compliance matrix, architecture, source attribution, phase log and verification receipts |
| `experiments/` | Independent attention/position verification and F01-F04 generators; later ablations pending |
| `notebooks/` | Reserved for a portable, actually demonstrated Colab notebook |
| `api/` | Reserved for future FastAPI and hosting packaging |
| `report/evidence/` | Small F01/F02/F04 PNGs, F03 runtime text and hash receipts; final article pending |
| `data/` | Dataset provenance plan; actual datasets ignored |
| `checkpoints/` | Future weights and metadata, ignored |
| `outputs/{figures,logs,samples}/` | F01-F04 generated figures/shape log; model samples remain future; ignored |
| `.github/workflows/ci.yml` | Minimal CPU installation, tests, formatting and build checks |

Future pipeline: training-only tokenizer -> shifted batches -> token embeddings +
own sinusoidal positions -> own causal attention/MLP/residual/normalization blocks
-> vocabulary logits -> next-token loss -> autoregressive samples.
No empty attention/training functions are presented as working implementations.

## Milestone roadmap

Each phase ends with evidence and requires explicit approval for the next phase.

1. **Phase 1:** foundation, configuration, test infrastructure and complete compliance plan.
2. **Phase 2 / M1:** manual three-token attention calculation, scaled dot-product attention,
   causal masking, correctness tests and F01/F02; also sequential/parallel-depth explanation.
3. **Phase 3 / M2:** explicit multi-head split/merge and sinusoidal positions, F03/F04.
4. **Phase 4 / M3:** decoder/GPT assembly, diagram/parameter count and MLM-vs-CLM demo.
5. **Phase 5 / M4:** provenance-checked data, shifted targets, training/checkpoint engine,
   and a bounded tiny-overfit debugging gate.
6. **Phase 6 / M5:** separately approved hardware-qualified from-scratch training,
   checkpoint retrieval, real losses, five fixed-prompt samples and trained attention.
7. **Phase 7 / M6:** separately approved three controlled ablation comparisons.
8. **Phase 8:** trained-model CLI, FastAPI, approved online hosting and actual Colab demo.
9. **Phase 9:** ten-section Medium article, ten evidence figures, two observed limitations,
   class presentation and a human recorded demo.
10. **Phase 10:** clean-environment full acceptance and authorized submission/publication.

Dependencies and concrete acceptance gates are in [architecture](docs/architecture.md).
Future training and inference commands will be documented when implemented and tested.

## Data, reproducibility and responsible use

Future dataset: [Hugging Face Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare),
the link embedded in the handout. No dataset has been downloaded for this phase.
See [data plan](data/README.md) for license/source, hashing and split checks before use.
Do not select hyperparameters using the held-out test split.

No pretrained language-model checkpoints or high-level GPT/BERT architecture classes
are permitted. No implementation, configurations or experimental artifacts from another
student's assignment are used. All future adapted resources must be attributed and
understood by the team; [sources](docs/sources.md) tracks influences.
Generated text will be experimental, not factually reliable. Use only authorized data,
keep secrets and large artifacts out of Git, and review the final submission personally.

No service credentials are required. `.env.example` documents `PYTHONHASHSEED` (set
before starting Python) and `CUBLAS_WORKSPACE_CONFIG` (future CUDA determinism).
The application does not automatically load `.env`. `seed_everything` seeds Python,
NumPy and the CPU PyTorch generator and can explicitly qualify CUDA/MPS later;
unsupported devices raise errors. Repeatability is scoped to the same hardware/software,
not guaranteed across PyTorch releases or platforms.

Phase results, limitations and approval boundaries are in [phase log](docs/phase_log.md).
The detailed Phase 2 verification/publication receipt is in
[completion report](docs/phase2_completion_report.md). Phase 3's report is linked above.
Phase 3 PR merge and Phase 4 execution each require separate explicit approval.
