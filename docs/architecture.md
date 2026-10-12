# Architecture and approval gates

This is an independent Zaid assignment; the only existing tracked input was a minimal
README at `1ba89cba2aa04c355bedd16b70b3e07f1818553e`. No prior scientific artifacts
are imported. Phase 1 implements infrastructure only. The installed `src/` package
layout supports imports outside the repository without implicit `PYTHONPATH` changes.

## Proposed baseline decisions

| Setting | Proposal | Reason / what must be finalized |
| --- | --- | --- |
| Tokenizer / vocabulary | Character / unresolved `null` | Transparent mapping; fit training text in Phase 5; derive vocabulary and unknown/special-token policy |
| Context | 128 | PDF starting point; attention cost grows quadratically; qualify memory and later compare 64/128/256 |
| Width | 128 | Compact PDF starting point; parameter count waits for vocabulary and real model |
| Heads | 4 | Head dimension 32; independent projection/split/merge in Phase 3 |
| Layers | 4 | PDF starting point; no learned-capacity claim |
| FFN / activation | 512 / GELU | 4x expansion and recommended smooth nonlinearity |
| Positions | Sinusoidal | Explicit handout requirement; own formulas in Phase 3 |
| Dropout | 0.1 | Recommended starting range; future tiny-overfit debug may explicitly use zero |
| Optimizer / rate | AdamW / 0.0003 | Handout suggestion; weight decay/schedule need reviewed protocol |
| Batch / steps | 16 / 1,000 | Resource-sensitive placeholders; not training adequacy or timing claims |
| Temperature / length | 0.8 / 80 | Handout demonstration defaults; positive finite temperature; greedy separate |
| Device / seed | CPU / 42 | Bounded infrastructure checks; qualify hardware before future runs |
| Outputs | Relative `outputs` Path | Resolve from invocation directory; no creation on config load |

No config authorizes a run. `vocab_size=null` is a valid proposal; model compatibility
requires a resolved positive vocabulary matching the tokenizer. Positive integers exclude
booleans and coerced strings. Width must divide by heads. Dropout is finite in [0,1),
not limited to suggested 0-0.1. LR/temperature finite and positive; seed integer [0,2**32).
Optimizers: adamw/adam/sgd; future engine must dispatch/test all three before claiming
execution. Devices: cpu/cuda/mps, hardware checked on seed setup. Tokenizer: character;
positions: sinusoidal; activations: GELU/ReLU. Unknown/duplicate JSON fields, nonobjects,
malformed JSON, blank paths, vocabulary mismatch and excessive sequence length fail.

## Planned implementation boundaries

Future modules in the same package: `attention.py` (Phase 2), `positions.py` and
multi-head attention (Phase 3), `transformer_block.py`/`gpt.py` (Phase 4),
`tokenizer.py`/`dataset.py`/`train.py`/`checkpoint.py` (Phase 5), `generate.py`
(Phase 6 foundations / Phase 8 CLI). Experiments/API/notebooks share this one engine.
Proposed pre-LN organization must be finalized and justified in Phase 4.

Checkpoint plan: atomic writes; own state dict, full config, fitted tokenizer,
optimizer, step, RNG states, source commit, dataset hash/split, software/hardware
metadata. Preserve best-validation/final separately. Validate trusted artifacts and
fresh-process reload; large weights out of Git. Approved Drive/release retrieval
must be independently verified by size/hash and loading.

## Future phase dependencies and acceptance

| Phase | Dependency / authorization | Acceptance condition |
| --- | --- | --- |
| 2 | Phase 1 and explicit approval | Three-token dk=dv=2 derivation with every operation and human review; independent reference; attention shape/softmax/mask/causality/gradient tests; F01/F02; reasoned depth/memory discussion |
| 3 | Passing Phase 2 and approval | Independent per-head projections, explicit split/concat/output trace; mask broadcast tests; own dtype/device/offset-aware sinusoidal formulas; F03/F04 |
| 4 | Passing Phase 3 and approval | Own decoder/GPT, residual/normalization/dropout/FFN tests and future invariance; real parameter breakdown/diagram F05; MLM/CLM matrices and toy comparison |
| 5 | Passing Phase 4 and approval | Authorized corpus/license/hash, disjoint splits, train-only tokenizer and shifted windows; bounded tiny-overfit loss decrease; logs and atomic checkpoint/RNG reload |
| 6 | Passing Phase 5; exact run/account/hardware/budget/storage approval | Frozen config, actual hardware/time/loss logs, validation-only selection, five fixed-prompt outputs and real trained attention F06-F08; retrievable checkpoint |
| 7 | Genuine baseline; exact matrix/budget approval | Three controlled factor comparisons; common corpus/splits/tokenizer/eval; measured loss/params/time, head quality and context memory/throughput; confounds/limitations; F09 |
| 8 | Own trained checkpoint; host/account/cost approval | Clean-shell CLI, FastAPI load-once/input/error tests, actual online health/generate calls, portable running Colab; F10 plus hosting evidence |
| 9 | Genuine Phases 2-8 evidence; human authorship review | Ten-section article/archive/citations and ten figures; two observed failures; class deck; 5-7 minute script within 5-10 minute human recording; publication/recording verified separately |
| 10 | Complete artifacts; publication/submission approval | Fresh environment install/tests/tiny-overfit/checkpoint/three prompts/causality/README/security; all links/checklist verified; unresolved human tasks reported |

Stop after every phase. Phase 1 grants no permission for later training, ablations,
deployment, article publication, recording, merge or final submission.

## Resource and evidence policy

Review data rights/version before download (`data/README.md`); qualify hardware before
training. Preserve failed/partial runs. Never use test for model selection. Comparisons
use identical held-out text with conditioning/exposure budgets recorded. Optional
BPE/RoPE/learned positions/weight tying cannot replace mandatory components. Figures
retain generating commands, source commit and checkpoint identity where relevant.

Hosting plan: FastAPI `/health` and `/generate`, bounded prompts/output/concurrency,
controlled CORS, own model loaded once, no private prompt logging by default. Review
CPU/RAM/latency/artifact access/cost/cold starts, then deploy with approval and verify
live HTTP. Colab must work from a fresh runtime with authorized accessible artifacts.
Local scaffolding does not prove hosting.
