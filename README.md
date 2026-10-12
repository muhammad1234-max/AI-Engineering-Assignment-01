# Zaid's AI Engineering Assignment #01

Build a small LLM from scratch: Transformer Deep Dive.
Owner: **zaidbinnaveed**. Authorized repository:
https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.

The objective is an independently implemented, understandable decoder Transformer
trained on Tiny Shakespeare, with numerical attention derivations, controlled
experiments, and usable inference. The academic specification is `A1_AIE.pdf`
(11 pages); its source identity is recorded in [requirements](docs/assignment_requirements.md).

**Current status: Phase 1 foundation only.** Configuration validation, package imports,
CPU seed controls, and configuration inspection work. Attention, a model, a tokenizer,
training, inference, deployment and completed experiments are not yet implemented or claimed.
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

CPU-only PyTorch is intentional for Phase 1. CUDA dependencies and hardware/run
qualification belong to an explicitly approved future phase. If a platform has no
compatible wheel, report the install error rather than running an unapproved GPU job.
Direct dependencies are pinned in `requirements.txt` and `pyproject.toml`; the
tested full Windows environment is recorded in `docs/environment-windows-py314.txt`.

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
| `src/zaid_lm/` | Installed Python package: configuration, seed utilities, configuration CLI |
| `configs/` | Validated baseline proposal |
| `tests/` | Phase 1 behavioral tests; architecture tests added in later phases |
| `docs/` | Compliance matrix, architecture, source attribution, phase log and verification receipts |
| `experiments/` | Reserved for evaluation, attention inspection and controlled ablations |
| `notebooks/` | Reserved for a portable, actually demonstrated Colab notebook |
| `api/` | Reserved for future FastAPI and hosting packaging |
| `report/` | Reserved for report/article, class presentation and recording plan |
| `data/` | Dataset provenance plan; actual datasets ignored |
| `checkpoints/` | Future weights and metadata, ignored |
| `outputs/{figures,logs,samples}/` | Future genuine evidence; generated files ignored |
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

Phase 1 results, limitations and the stop condition are in [phase log](docs/phase_log.md).
