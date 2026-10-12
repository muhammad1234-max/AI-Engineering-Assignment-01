# Phase 2 verification and completion report

Phase 2 engineering deliverables are complete. The academic M01 gate still requires
Zaid's personal mathematical review/understanding; no such review is claimed here.
This report records actual evidence, not trained model results. Publication and
completed CI receipts are below. Phase 3 requires separate authorization.

## Identity and integrated starting point

Owner/authenticated account: `zaidbinnaveed`. Workspace:
`C:\Users\Zaid\OneDrive\Desktop\AI-Engineering-Assignment-01`.
Origin fetch/push: `https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.git`.
PR1's reviewed head `25c5f627580faa2e144173b6013aac2fb858b053` and all four
successful checks were reverified. The new Phase 2 handoff explicitly authorized
its merge. [PR1](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/pull/1)
was merged at **2026-10-12 13:38:23 UTC** using `--merge --match-head-commit`.
Resulting main and Phase 2 start: `f04dd07cb8c5fad8cd21ae19f111b548283706f8`.
Fetched origin, fast-forwarded local main, created `phase-2-attention-engine`.

All 11 PDF pages were text-extracted and visually reviewed again. The source PDF
and historical Phase 1 handoff/master remained unchanged. The master is Version 1.0;
the historical handoff's Version 1.1 label has no supplied corresponding document.
The new Desktop handoff is preserved byte-for-byte in `docs/phase2_instructions.txt`
(SHA256 `52ede30cb7d4c397c799470969c37fb6f2712e138e97d109ca41670e5e413f20`).
No unrelated assignment implementation, checkout or evidence was consulted/copied.

## Added and modified inventory

Added:

* `src/zaid_lm/attention.py`
* `tests/test_attention.py`
* `experiments/__init__.py`
* `experiments/attention_example.py`
* `experiments/visualize_causal_mask.py`
* `docs/manual_attention.md`
* `docs/why_transformers.md`
* `docs/phase2_instructions.txt`
* `docs/environment-windows-py314-phase2.txt`
* `docs/phase2_completion_report.md`
* `report/evidence/F01_manual_attention.png`
* `report/evidence/F01_manual_attention.json`
* `report/evidence/F02_causal_mask.png`
* `report/evidence/F02_causal_mask.json`
* `report/evidence/attention_verification.json`

Modified:

* `.gitattributes` (preserve new handoff bytes)
* `.github/workflows/ci.yml` (attention/evidence checks added)
* `pyproject.toml`, `requirements.txt` (Matplotlib 3.10.8 development pin)
* `scripts/verify_repository.py` (only two named small PNG exceptions, source/content hashes)
* `README.md`
* `docs/assignment_requirements.md`
* `docs/phase_log.md`
* `docs/sources.md`

Phase 1 configuration, reproducibility source, CLI, original tests, baseline JSON and
historical instruction copies are preserved. Generated output PNGs, environment,
build/dist, caches and PDF-review renders remain ignored; only the two small validated
report PNGs are tracked. No ignore-policy override for arbitrary binary artifacts.

## Actual mathematical example and results

Exactly three tokens, d_k=d_v=2. These are supplied numeric representations,
not learned projections or trained embeddings. All matrices below use six decimals;
the [derivation](manual_attention.md) contains exact expressions and every required
arithmetic step. Compute with unrounded numbers.

```text
Q = K = [[1,0], [0,1], [1,1]]
V = [[1,2], [3,0], [0,4]]
QKᵀ = [[1,0,1], [0,1,1], [1,1,2]]

Unmasked scaled S (3×3):
[[0.707107, 0.000000, 0.707107],
 [0.000000, 0.707107, 0.707107],
 [0.707107, 0.707107, 1.414214]]

Unmasked A (3×3):
[[0.401112, 0.197776, 0.401112],
 [0.197776, 0.401112, 0.401112],
 [0.248255, 0.248255, 0.503490]]

Unmasked O (3×2):
[[0.994440, 2.406673],
 [1.401112, 2.000000],
 [0.993020, 2.510470]]

Additive causal mask M (3×3):
[[0, -inf, -inf], [0, 0, -inf], [0, 0, 0]]

Masked S+M (3×3):
[[0.707107,     -inf,     -inf],
 [0.000000, 0.707107,     -inf],
 [0.707107, 0.707107, 1.414214]]

Masked A_c (3×3):
[[1.000000, 0.000000, 0.000000],
 [0.330238, 0.669762, 0.000000],
 [0.248255, 0.248255, 0.503490]]

Masked O_c (3×2):
[[1.000000, 2.000000],
 [2.339523, 0.660477],
 [0.993020, 2.510470]]
```

Independent scalar-loop verification covers all seven intermediates, not only final
output. The production engine differs by at most **1.1102230246251565e-16** on this
float64 example. The nine-decimal document appendix differs by at most
**4.845538414954831e-10**, within the rounding-aware 5.1e-10 absolute tolerance.
Engine comparison uses rtol=atol=1e-12. Full results/errors are in
`report/evidence/attention_verification.json`; JSON null explicitly denotes -inf.
The scalar oracle uses independent Python loops/index tests and stable `math.exp`,
without production attention/masking/softmax helpers. Torch values were corroboration,
not pasted as a purported hand calculation. Random/different-width examples also match.

## Public attention contract

```python
causal_forbidden_mask(length: int, *, device=None) -> torch.Tensor
scaled_dot_product_attention(q, k, v, *, causal=False, mask=None) -> (output, weights)
```

Q `(..., Tq, dk)`, K `(..., Tk, dk)`, V `(..., Tk, dv)`; identical leading
dimensions (batch and optional head axes), positive dimensions, finite float32/float64
inputs, identical dtype/device. Output `(..., Tq, dv)`, weights `(..., Tq, Tk)`.
No leading-input broadcasting; the mask broadcasts exactly to the weight shape.
Unmasked non-square attention is supported. Causal attention requires equal lengths
and same-index alignment, diagonal allowed; there is no cache offset convention.

True means forbidden for boolean masks. Additive masks must have the input floating
dtype and entries exactly 0/-inf. Explicit/causal masks combine by union before softmax.
All-blocked rows are rejected, not converted to NaN/uniform/zero fabricated distributions.
Shape/device mismatches, empty axes, nonfinite inputs, overflowing computed scores/output,
unsupported precision and invalid mask entries produce clear ValueError/TypeError.
Float16/bfloat16/complex/integer inputs and arbitrary finite additive biases are outside
this intentional Phase 2 contract. No dtype conversion or silent device fallback.

The visible core is matmul -> divide by sqrt(dk) -> masked_fill(-inf) ->
`torch.softmax(dim=-1)` -> matmul with V. Autograd remains intact; input tensors are
not mutated. No learned Q/K/V projection, dropout, head split/merge or fused attention.
Finite-value validation uses eager scalar checks; this correctness-first engine is
not claimed to be a compiled/high-throughput GPU kernel. Only CPU was verified.

## Local verification

Windows x64, CPython 3.14.0, torch 2.10.0+cpu, NumPy 2.4.6, pytest 9.0.3,
Ruff 0.15.4, Matplotlib 3.10.8. The full installed environment is recorded separately.
Commands ran from the authorized repo; `PY` below abbreviates
`.\.venv-phase1\Scripts\python.exe`.

| Executed verification | Actual result |
| --- | --- |
| Original suite before implementation: `PY -m pytest -q` | 54 passed in 66.66s |
| New tests: `PY -m pytest tests/test_attention.py -q` | 82 passed in 3.61s after corrections |
| Complete editable suite: `PY -m pytest -q` | 136 passed in 23.03s; zero failed/skipped |
| `PY -m ruff check .`, `PY -m ruff format --check .` | Passed; final publication checks recorded below |
| `PY -m pip check` | No broken requirements |
| `PY -m pip install --no-build-isolation --no-deps -e .` | Passed |
| `PY -m build --no-isolation` | sdist and wheel built successfully |
| `PY -m pip install --no-deps --force-reinstall dist/zaid_lm-0.1.0-py3-none-any.whl`, then full pytest | Installed noneditable wheel; all 136 passed in 186.34s; zero failed/skipped |
| `PY -m experiments.attention_example --figure` | Seven-matrix scalar/manual/engine check PASS; F01 generated |
| `PY -m experiments.visualize_causal_mask` | 3/8-token index-rule check PASS; F02 generated |
| Visual inspection of both PNGs after regeneration | Correct labels, dimensions, matrices, mask convention; no overlapping text |

Tests cover batch/head independence, row normalization, feature-width scaling,
variable/value widths, non-square attention, singleton, exact zero forbidden weights,
future perturbation invariance, nonzero finite Q/K/V gradients, float64 finite-difference
gradcheck (causal/unmasked), exactly zero future key/value gradient influence, duplicates,
zero/negative/mixed-sign/large finite logits (float32 and float64), mask broadcasting and
union, explicit dtype/device/shape/nonfinite/overflow/fully-blocked/error policies.
Device mismatch uses a meta tensor to exercise validation without running a GPU.
Tiny gradchecks set one CPU thread temporarily and restore the previous thread count.
All original 54 tests are unchanged and included in 136.

## F01/F02 evidence and conceptual foundations

Exact commands (repository root):

```powershell
.\.venv-phase1\Scripts\python.exe -m experiments.attention_example --figure
.\.venv-phase1\Scripts\python.exe -m experiments.visualize_causal_mask
```

| Figure | Ignored generation location | Tracked report evidence | Bytes / SHA256 |
| --- | --- | --- | --- |
| F01 | `outputs/figures/F01_manual_attention.png` | `report/evidence/F01_manual_attention.png` | 127,047 / `dc94827d16f1a70c9d35b0f64a348c1a9e39afb2b1147985ac7f2bd37af4af09` |
| F02 | `outputs/figures/F02_causal_mask.png` | `report/evidence/F02_causal_mask.png` | 76,383 / `a9c5bdbabbe6e8cf61b3b34d3079984066cac36c9e8a442a3c675d436a57b2c0` |

Neighboring JSON receipts hash exact generator/attention/derivation sources and PNG bytes.
F01 is a verified typed calculation summary, not handwriting. F02 is structural
visibility; F07 trained attention remains future work. Fonts/platforms may change
rendered pixels; hashes identify this evidence, not a cross-platform pixel guarantee.

`docs/why_transformers.md` explains recurrence, teacher-forced parallel causal training,
layer/position depth, sequential generation, legal distant connections, dense quadratic
attention memory, projection/FFN costs and hardware dependence. Its table explicitly
states assumptions. A bounded MAC example uses T=d=128, f=512, B=1 and is reasoned,
not timed or evidence of model quality. No universal faster/lower-memory claim is made.

## Requirements, errors and limitations

L01 and structural F02 are VERIFIED. Pure attention, complete typed arithmetic,
gradient/mask tests and figure engineering are VERIFIED in the added acceptance table.
H01-H06/F01/L02/M01 retain IN_PROGRESS for personal arithmetic review; H07 remains
HUMAN_ACTION. I04's learned projections await Phase 3; I09's full-model integration
awaits Phase 4. No human attestation or academic final completion is claimed.

Corrected issues:

* Sandbox dependency lookup initially said no Matplotlib distribution; approved network
  retry installed the compatible cp314 wheel. GitHub reads also required approved access.
* Initial Ruff check caught compact semicolon statements; formatter expanded them and
  final lint passed. One patch missed reformatted context, applied no partial edits,
  and was reissued against the actual file.
* Two tests incorrectly created expected 1/3 in float32 before comparing as float64.
  Fixed expected dtype; both pass without weakening the float64 tolerance.
* Initial default CPU thread count slowed tiny gradchecks; scoped single-thread fixture
  speeds checks and restores settings. This is not a model timing benchmark.
* Initial F02 footer/legend overlapped; reserved layout space and reinspected regenerated
  figure. F01 row/column labeling was improved before retaining its final evidence.
* `gh pr diff --stat` was unsupported; reviewed `--name-only` and local diffstat instead.

Remaining limits: human mathematical review; CPU-only verification; eager validation;
no half precision or causal cache alignment; no trained model or speed measurements.
Historical master version discrepancy is documented. Dataset rights/runtime/hardware
qualification remain later gates; no corpus download, training or deployment started.

## Phase 3 recommendation and stop

After separate approval: implement learned Q/K/V and output projections in multi-head
attention, explicitly split/permute heads, call this verified core per head, then
concatenate/merge/project; add shape/reference/mask/gradient tests and genuine F03.
Implement own sinusoidal positions with formula/shape/dtype/device/determinism/offset
tests and genuine F04. Preserve configuration and existing tests. Full blocks, tokenizer,
data, real training, ablations, generation and API remain in their later phases.

**Stop after Phase 2 publication and verified CI; PR2 merge and Phase 3 need approval.**

## Verified publication and final engineering receipt

Implementation commit: `cfc7db3d9a986d0640b3c97648f88ab24a423877` on
`phase-2-attention-engine`, pushed to the exact authorized origin.
[PR2](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/pull/2) is OPEN,
base `main` at `f04dd07cb8c5fad8cd21ae19f111b548283706f8`. No Phase 2 merge.
Final delivery adds only this report/log receipt; its exact final commit is provided
in the delivery message and can be resolved with `git rev-parse phase-2-attention-engine`
or PR2's headRefOid. A commit cannot embed its own resulting SHA.

| Actual CI run on implementation commit | Python 3.11 | Python 3.14 | Result |
| --- | --- | --- | --- |
| [Push 38260385925](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38260385925) | 136 passed in 2.07s | 136 passed in 2.66s | COMPLETED SUCCESS |
| [PR 38260453436](https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38260453436) | Job completed successfully | Job completed successfully | COMPLETED SUCCESS |

CI job step receipts were inspected, not inferred from submission: dependency
installation/check, all tests, Ruff lint/format, index audit, configuration CLI,
seven-matrix verification/F01, F02 generation and sdist/wheel build all SUCCESS on
both Python versions. The workflow display name initially retained the older cached
"Phase 1 foundation" name; actual new attention/audit/figure steps were verified.

After installed-wheel validation, restored the editable installation successfully;
final lint passed and format reported 13 formatted files. Regenerated F01/F02 again:
the PNG SHA256 values exactly matched the retained report evidence, and `git status
--short` remained empty. This establishes repeatability in this Windows environment,
not cross-platform pixel identity. Final index audit: PASS, 49 files, largest 127,047
bytes, no findings. Source config/reproducibility/original tests/handoffs compared
against integrated main and have no changes. Working branch tracks the pushed origin.

The final receipt commit's CI is checked before delivery; any result reported in the
delivery message refers to that exact final head. Phase 2 engineering is complete;
personal H07/M01 review and authorization for PR2 merge/Phase 3 remain outstanding.
