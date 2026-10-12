# Phase 3 verification and completion receipt

Local engineering validation is complete. Publication and Linux CI results will be
recorded below once they have actually finished. Only Phase 3 is implemented.

## A. Repository integrity

* Authorized repository: https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01.
* Actual authenticated GitHub account: `zaidbinnaveed`; push/maintain/admin permission
  verified before implementation. Origin is exactly the authorized HTTPS repository.
* Local workspace: `C:\Users\Zaid\OneDrive\Desktop\AI-Engineering-Assignment-01`.
* Reviewed PR2 head `bf9df03a55c16b7dc0018cd4d268c35900f4b336` remained unchanged,
  with all four push/PR Linux 3.11/3.14 checks successful and no merge conflict.
* Explicit user authorization: **"Authorize the merge and complete Phase 3"**.
  PR2 was merged with `--merge --match-head-commit bf9df03...` at
  `2026-10-12T14:26:59Z`. Integration/Phase 3 starting commit:
  `3b8bcd3df0c0d456694b2e6deaeaa09b6ab7b0cd`.
* Main was fast-forward synchronized; the Phase 2 source/tests/config/manual/evidence
  tree was compared with the reviewed head and unchanged. All original 136 tests
  then passed before coding. Feature branch: `phase-3-multihead-position`.
* Final implementation/publication SHA and CI receipt: pending below. The final
  delivery message supplies the final documentation head, which cannot be embedded
  in its own commit. No main history rewrite or force-push is used.
* No unrelated project assets, code, configurations, results or checkpoints reused.
  The original 11-page PDF and master specification were reread; current instructions
  are preserved byte-for-byte with SHA256
  `e958c332c59ddc7ddfea8906d2c9514eee70839feace5f8aa9cc26126c9bc217`.

## B. Implementation and complete changed-file inventory

| File | Change |
| --- | --- |
| `src/zaid_lm/multihead_attention.py` | Four independent trainable linears, explicit head operations, exact Phase 2 reuse, inspection and output dropout |
| `src/zaid_lm/positional_encoding.py` | Own fixed sinusoidal formula, persistent buffer, bounded offsets, addition and dropout |
| `experiments/multihead_shape_trace.py` | Captured real runtime shapes; Python-loop linear/head/output oracle; parameter, causal and gradient checks |
| `experiments/visualize_positions.py` | Full independent baseline scalar comparison, heatmap and frequency curves |
| `experiments/phase3_evidence.py` | Small text/PNG copies with content/source provenance |
| `tests/test_multihead_attention.py` | Numerical, axes, trainability, mask, dropout and invalid-contract checks |
| `tests/test_positional_encoding.py` | Even/odd/width-one scalar formulas, state, boundaries, dtype/device and dropout checks |
| `tests/test_phase3_integration.py` | Position-plus-attention shape/gradient/causal/state/permutation checks |
| `report/evidence/F03_multihead_shape_trace.txt` | Actual executed shape/numerical/gradient trace |
| `report/evidence/F03_multihead_shape_trace.json` | Exact text/source hashes and reproduction command |
| `report/evidence/F04_positional_encoding.png` | Implementation-derived baseline position figure |
| `report/evidence/F04_positional_encoding.json` | Exact PNG/source hashes, numerical errors and reproduction command |
| `docs/phase3_instructions.txt` | Exact authorized Phase 3 specification |
| `docs/multihead_attention.md` | Mathematics, APIs, axis rationale, diagram, masks, parameter counts and limitations |
| `docs/positional_encoding.md` | Formula, addition/buffer/offset policy, numerical example, symmetry and F04 interpretation |
| `docs/phase3_completion_report.md` | This A-F engineering and publication receipt |
| `docs/phase_log.md` | Authorization/integration, commands, results, issues, CI and stop boundary |
| `docs/assignment_requirements.md` | Only evidence-backed Phase 3 rows VERIFIED; future/human gates preserved |
| `docs/architecture.md` | Actual module boundaries and reviewed Phase 3 decisions |
| `docs/sources.md` | Primary equation/buffer attribution and original-work provenance |
| `README.md` | Current status, actual commands, module contracts and evidence links |
| `.gitattributes` | Preserve Phase 3 instruction bytes/hard-break spaces and F03 LF artifact bytes |
| `.github/workflows/ci.yml` | Add F03/F04 execution and installed-wheel suite on both supported CI versions |
| `scripts/verify_repository.py` | Exact F04 PNG allowance and F03 text/source receipt verification |

The original engine, 136 older tests, baseline configuration, Phase 2 derivation,
F01/F02 sources and evidence are unchanged. No new runtime dependency is introduced.

`MultiHeadSelfAttention(D,H,dropout=0.0,bias=True,*,device=None,dtype=None)` supports
`forward(x,*,causal=True,mask=None,return_attention=False)`. Default returns `(B,T,D)`
as one tensor; inspection returns `(output,weights)` with weights `(B,H,T,T)`.
Q/K/V project `(B,T,D)` independently, reshape `(B,T,H,Dh)`, permute `(B,H,T,Dh)`;
engine scaling is sqrt(Dh). Head outputs permute back, become contiguous, reshape
`(B,T,D)` and receive the fourth learned linear. D is divisible by H, equal key/value
width Dh is D/H. Four biased full-width projections give `4*(D²+D)=66,048` actual
baseline parameters; bias-free `4*D²=65,536` is also tested. This is not GPT's count.

Masks keep Phase 2 conventions: True forbids, same-dtype 0/-inf additive masks,
broadcast to `(B,H,T,T)`, union with causal j>i, all-forbidden rows rejected.
Batch masks use `(B,1,T,T)`. Output-only dropout preserves normalized returned
weights and disables in eval. Dropout accepts finite numeric p in [0,1).

`SinusoidalPositionalEncoding(D,context_length,dropout=0.0,*,device=None,dtype=float32)`
adds positions with `forward(x,*,offset=0)`. Even features are
`sin(pos/10000^(2i/D))`; odd features are cosine at the same frequency. Odd widths
retain the final sine. A persistent `(1,context,D)` buffer supplies `(1,T,D)` slices,
preserves `(B,T,D)` on addition, has zero trainable parameters and migrates/serializes
normally. Offset is an integer >=0, excluding bool; offset+T must fit context.
There is no KV cache. Both modules require matching finite float32/64 input and
module dtype/device. PE computes in float64 before storage; construct directly
with float64 for double precision, since casting stored float32 cannot restore digits.
Positional dropout follows addition and disables in eval. No input is mutated.

## C. Actual numerical validation

Actual CPU float32 runtime trace, seed 42, evaluation and zero dropout:

```text
X and Q/K/V:              (2,8,128)
Q/K/V split:             (2,8,4,32)
Q/K/V permuted:          (2,4,8,32)
Per-head scores/weights: (2,4,8,8)
Head outputs:            (2,4,8,32)
Merged/final projection: (2,8,128)
Parameter count: 66048 PASS
Future perturbation final prefix max error: 0.0 PASS
All future probabilities: exactly zero PASS
```

The full F03 text records each tensor-derived shape individually. Projections and
merged input are hooked from the production forward; head inspection calls the
unchanged engine on these captured tensors and asserts exact agreement. Scores are
separately computed for inspection, not attributed to the engine's return contract.

The fixed asymmetric B=1,T=3,D=4,H=2 example uses independent Python loops for
all projections, per-head feature slicing, Phase 2 scalar attention, concatenation
and output projection. No multi-head forward/matmul/head helper is in the oracle.

| Mode | Output max absolute error | Weight max absolute error | Tolerance |
| --- | --- | --- | --- |
| Unrestricted float64 | 2.220446049250313e-16 | 2.220446049250313e-16 | rtol=atol=1e-12 |
| Causal float64 | 8.881784197001252e-16 | 1.1102230246251565e-16 | rtol=atol=1e-12 |

For example the first causal output row is `[0.69,-1.585,1.475,4.285]` up to
double rounding. The full three-row outputs are retained unrounded in F03.
Sentinel slices prove head/token feature order; the asymmetric reference detects
direct-reshape mistakes. Bias-on/off, 1/2/4/8 heads and batch references also pass.

Actual baseline weight gradient norms from the bounded squared-output backward:
Q `0.010663215070962906`, K `0.01168854907155037`, V `0.1009291261434555`,
O `0.07194755226373672`. Tests also check finite nonzero gradients for input and
all four weights/biases; input gradcheck passes with eps=1e-6, atol=1e-5, rtol=1e-3.
Prefix perturbations cover every cutoff in small sequences, both dtypes, and
position-plus-attention with fixed positional indices. Expected error is exactly
zero in these CPU checks. Batch independence uses 1e-12 float64 tolerances.

All 16,384 baseline position values independently match scalar sine/cosine:
float32 max absolute error `2.979937197000737e-08` (atol=1e-6), constructor float64
`4.440892098500626e-16` (atol=1e-12). Values range [-1,1]; position zero is exactly
alternating zero/one. Odd widths 1/3/7, pair identity, offsets, persistent state,
identity addition gradient and repeated eval calls are verified. Unmasked attention
is permutation-equivariant; adding fixed slot positions breaks that symmetry for
reordered token content. A fixed causal mask already encodes order; docs state this.

## D. Evidence and provenance

* F03: `report/evidence/F03_multihead_shape_trace.txt`, **2,272 bytes**, SHA256
  `733986d973544c163d4b48630a2bc527650f03cd4dee217cde6c2d3150510ba5`.
  Command: `python -m experiments.multihead_shape_trace`. Proves executed baseline
  dimensions, engine inspection agreement, parameter count, causal output and finite
  projection gradients; also records independent small-reference results.
* F04: `report/evidence/F04_positional_encoding.png`, **313,423 bytes**, SHA256
  `f56321ea799afca8612c816d2f79fa5984ffcea269c882337669597c5769ef04`.
  Command: `python -m experiments.visualize_positions`. Proves the implemented fixed
  baseline position structure, feature frequencies/pairing and value bounds after
  scalar verification. Heatmap axes/colorbar and selected-dimension curves were
  visually inspected. This is not trained attention or a trained-model result.
* Neighboring JSON receipts contain every generating source hash and exact command.
  F03/F04 receipts are verified against Git-index content/source bytes by the audit.
  PNG bytes may differ across fonts/platforms; CI verifies committed provenance before
  regenerating local evidence. Raw outputs are ignored; only reviewed small copies
  are tracked. Existing F01/F02 receipts remain unchanged.
* F05-F10 remain future architecture/training/attention/sample/ablation/inference
  evidence. F01's personal arithmetic review remains pending despite verified code.

## E. Engineering verification

Local Windows x64 Python 3.14.0; CPU torch 2.10.0, numpy 2.4.6, pytest 9.0.3,
ruff 0.15.4, matplotlib 3.10.8, build 1.6.1, setuptools 84.0.0. No GPU work.

| Executed command (using `.venv-phase1/Scripts/python.exe`) | Actual result |
| --- | --- |
| `-m pytest tests/test_config.py tests/test_foundation.py tests/test_attention.py -q` before coding | Original **136 passed in 98.54s** |
| `-m pytest tests/test_multihead_attention.py tests/test_positional_encoding.py tests/test_phase3_integration.py -q` | **105 passed in 44.33s**; final small-example bias precision refinement covered by subsequent full suite |
| `-m pip install --no-build-isolation --no-deps -e .` | Editable build/install successful |
| `-m pip check` | No broken requirements |
| `-m build --no-isolation` | sdist and `zaid_lm-0.1.0-py3-none-any.whl` built successfully |
| `-m pip install --no-deps --force-reinstall dist/zaid_lm-0.1.0-py3-none-any.whl` | Noneditable installed wheel successful |
| Import path assertion for installed wheel | `.venv-phase1/Lib/site-packages/zaid_lm/`; both new modules import |
| `-m pytest -q` against installed wheel | **241 passed in 16.04s**, zero failures/skips |
| `-m ruff check .` | PASS |
| `-m ruff format --check .` | All **21 Python files** formatted |
| Shape trace and position figure commands above | PASS, evidence actually generated and inspected |
| `git diff --check` | PASS; routine Windows text-normalization warning only |
| `scripts/verify_repository.py` | PASS: 65 tracked files, largest 313,423 bytes, zero findings; all evidence/source hashes match |

CI is pending publication, not claimed successful here yet. The staged audit passed.
CI matrix is Linux Python 3.11/3.14, read-only permissions, CPU dependency installs,
editable suite, lint/format, committed evidence audit, configuration inspection,
all F01-F04 verification/generators, build, noneditable wheel install and full suite.

Issues resolved: formatting required rerunning after a long example-bias line;
fixed example biases were explicitly constructed in float64 rather than cast from
float32 so their printed decimal values match the intended literal example. The
old engine/math/tests were unchanged. Supplied instructions contain five intentional
Markdown hard-break trailing spaces; a path-specific Git whitespace attribute preserves
them byte-for-byte while checking all implementation files normally. F03 text is forced
to LF so its artifact hash survives Windows/Linux checkout. No scientific test failed.

Remaining limitations: input-value checks add unprofiled overhead; no accelerator
qualification, mixed precision, cache or complete decoder exists. Context is bounded
by the precomputed table. Reproducibility is scoped to software/hardware; PNG pixels
can vary with plotting/fonts. Standard `.to()` casting does not recover discarded
precision. Human hand-calculation understanding is unverified and remains required.

## F. Next steps and explicit stop

Phase 3 implements L03/L04/M02/I03/I04/I05/F03/F04 only. I09 is still IN_PROGRESS
until full-model integration. H01-H07/L02/M01/F01 personal review gates remain open.
All remaining assignment rows are retained explicitly in `assignment_requirements.md`:
complete decoder/embeddings/FFN/residual/normalization/LM head and MLM-versus-CLM
(L05/L06/M03/I02/I06-I10/F05); tokenizer/data provenance/shifted targets and bounded
overfit (I01/I11/S03/S05-S07/M04); real training/checkpoint/evaluation and five samples
(L07/M05/T01-T08/I12/F06-F08); three controlled ablations (M06/E01-E05/F09);
trained CLI/REST hosting/Colab/retrievable artifacts (D02-D04/D09/F10); ten-section
article, two observed failures, presentation, recording, human understanding and
submission/final clean acceptance (D05-D08/R01-R10/AT01-AT03/V01-V08).

No current Phase 3 mathematical deficiency is known. Review PR3, then separately
authorize its merge and Phase 4. Recommended Phase 4 plan: finalize the proposed
pre-LN residual layout; integrate these modules with elementary LayerNorm, FFN,
token embedding and vocabulary logits; verify full decoder/GPT shapes/gradients,
causal prefix invariance, parameter breakdown/diagram and controlled toy MLM/CLM
visibility/objective comparison. Corpus download and language-model training remain
later approvals. This report proposes that plan without implementing it.

PR URL and successful final CI receipt will be recorded below. **Stop after Phase 3;
do not merge its PR or execute Phase 4 without separate explicit approval.**
