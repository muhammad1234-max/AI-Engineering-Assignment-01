# Phase 1 log

Historical Phase 1 receipt below is preserved. The later authorized Phase 2 execution
and PR1 merge are recorded at the end; earlier stop/open-PR statements describe
the Phase 1 endpoint, not the current repository state.

Phase 1: COMPLETE. Local implementation/verification and feature-branch publication
verified. Full assignment: NOT COMPLETE. Pull request open; no merge performed.
Date: 2026-10-12, Windows x64 / PowerShell, Python 3.14.0, CPU-only torch 2.10.0+cpu.

## Authorization, identity and preserved evidence

User instructed Phase 1 using Desktop instructions.txt and then said to continue.
The handoff explicitly authorizes the Phase 1 feature-branch commit, push and reviewable
pull request; no merge. Correct root:
`C:\Users\Zaid\OneDrive\Desktop\AI-Engineering-Assignment-01`.
Source main: `1ba89cba2aa04c355bedd16b70b3e07f1818553e` (Initial commit).
Existing dedicated branch `phase-1-foundation` was verified and used; main untouched.
`origin` fetch/push both point to the authorized zaidbinnaveed repository.

Executed: `Get-Location`, `git rev-parse --show-toplevel`, `git remote -v`,
`git status --short`, `git log -1`, `git branch -vv`, `git fetch origin main`,
`git rev-parse origin/main`, `gh auth status`, `gh api user --jq .login`,
`gh api repos/zaidbinnaveed/AI-Engineering-Assignment-01` with selected fields,
and `gh pr list --repo zaidbinnaveed/AI-Engineering-Assignment-01 --state open`.
Account **zaidbinnaveed**, permissions **admin/maintain/pull/push/triage=true**;
no open PRs at initial check. Initial sandbox network checks failed and gave a misleading
invalid-token message; outside-sandbox verification succeeded. Credentials never printed
unmasked or written to project files.

Before changes, only README was tracked; `.venv` existed untracked and its pip failed
with `SyntaxError: source code string cannot contain null bytes`. Preserved unchanged;
fresh `.venv-phase1` created instead. Previous incident checkout/evidence untouched.
No unrelated implementation, data, configs, checkpoints/results/reports or Git history
were imported. References to the revoked project in supplied instructions are exclusion
rules, not scientific attribution or reuse.

## Source review and planning

Read the entire Desktop Phase 1 handoff and independent Downloads master
`instructions (1).txt`. Older erroneous master was identified as revoked. Read all 11
PDF pages with PyMuPDF text extraction and rendered PNG inspection; diagrams/tables,
rubric and checklist reviewed. First extraction hit Windows console Unicode encoding;
rerun with `python -X utf8` succeeded. Extracted actual PDF hyperlink destinations.
Source PDF SHA256 unchanged:
`30e543d6d429f60bba96f7c20e2131d198551112b018520f6eae6dd8a23c6bdd`.
Renders are outside the project under Desktop `tmp/zaid-phase1-pdf`, never evidence figures.

Recorded the handoff's Version 1.1 vs available independent master's Version 1.0 mismatch.
Preserved exact authorized instruction copies in repository with binary Git attributes
to keep source bytes. Created complete matrix for all PDF sections, all seven hand steps,
configuration items, ten figures, rubric, ten report sections, acceptance and final boxes.
No future artifact is claimed complete. Sources/architecture/data plan document influences,
split/rights policy, checkpoint plan, hosting/Colab plans and dependencies/gates for Phases 2-10.

## Implementation

Created installed src-layout `zaid_lm` package with strict frozen dataclass/JSON config,
unknown/duplicate-field rejection, range/type/divisibility validation, unresolved-vocabulary
and context guards, inspection CLI, and actual Python/NumPy/PyTorch CPU seed setup.
CPU seed tests assert CUDA is not touched. Explicit CUDA/MPS paths report unavailable
devices rather than silently falling back; no accelerator execution was performed.
Added direct dependency pins, pytest, Ruff, safe ignores, env documentation and minimal
read-only CI for Linux Python 3.11/3.14. All future implementation directories are labeled
pending. No attention engine, model, dataset pipeline or training loop was implemented.

## Actual installation and verification receipts

Commands below ran from repository root unless noted. `PY` abbreviates
`.\.venv-phase1\Scripts\python.exe` only in this log; README has exact executable paths.

| Executed command | Actual result |
| --- | --- |
| `python -m venv .venv-phase1` | Fresh isolated env created; pip 25.2 / Python 3.14.0 |
| `PY -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 0` | Installed torch 2.10.0+cpu Windows cp314 wheel; sandbox lookup failed first, approved network retry succeeded |
| `PY -m pip install -r requirements.txt --timeout 30 --retries 0` | Passed after approved network retry; initial NumPy 2.4.0 pin was reported yanked for compatibility bug and replaced with 2.4.6 before final verification |
| `PY -m pip install --no-build-isolation -e .` | Editable zaid-lm 0.1.0 installation succeeded |
| `PY -m pip check` | No broken requirements found |
| `PY -m pytest -q` (editable install) | **54 passed in 13.36s**, zero failed/skipped |
| `PY -m ruff check . --fix` / `PY -m ruff format .` | Fixed import order and applied formatting before verification |
| `PY -m ruff check .` | All checks passed |
| `PY -m ruff format --check .` | Seven source/test files already formatted at initial check; eight including audit script at final check |
| `PY -m zaid_lm --config configs/baseline.json --device cpu` | Valid proposal_only JSON; no training or outputs created |
| `.\.venv-phase1\Scripts\zaid-lm-config.exe --config configs/baseline.json --seed 123` | Console entrypoint succeeded; printed overridden seed 123 |
| `PY -m build --no-isolation` | Built sdist and noneditable wheel successfully; ignored dist outputs |
| `PY -m pip install --no-deps --force-reinstall dist\zaid_lm-0.1.0-py3-none-any.whl` | Built wheel installation succeeded |
| `PY -m pytest -q` (wheel install) | **54 passed in 12.80s**, zero failed/skipped |
| Absolute PY import and module CLI from Desktop `tmp` | Imported actual site-packages wheel; configuration CLI worked outside repo |
| `PY -m pip install --no-build-isolation -e .` (restore development) | Succeeded; development env restored to editable install |
| `PY -m pip freeze --exclude-editable` | Captured 20 installed dependency versions in environment-windows-py314.txt |
| `git check-ignore` generated data/weights/logs/envs/secrets and retained placeholders | Generated files ignored; placeholders/.env.example unignored (expected exit 1 for latter query) |
| `git add .`, `PY scripts/verify_repository.py`, `git diff --cached --check` | Initial 33-file index audit PASS, largest 44,429 bytes, no findings; diff check passed. Final receipt updated after documentation |

No GPU build: `torch.version.cuda is None`; all behavioral tests ran on CPU. No training,
losses, generated model samples, benchmarks or figures are claimed. The wheel/virtual
environment checks above are Phase 1 infrastructure validation, not full-assignment
clean-environment acceptance. Phase 10's trained checkpoint/overfit/inference/hosting
acceptance remains pending.

## Security and provenance audit

Reviewed all created/modified sources, tests, config, manifests, workflow, documentation
and supplied instruction copies. Git index audit checks exact remote, file sizes/binary
content, common credential/private-key patterns, artifact extensions and prohibited
model-library imports; suspected values are never printed. This is a scoped scanner,
plus manual review, not a claim to prove absence of every possible secret format.
No actual data/checkpoints/GPU artifacts/secrets are staged. Ignore rules verified.
All requirement rows have appropriate phase dependencies/evidence; future claims pending.

## Outstanding items and stop

No Phase 1 blocker. GitHub publication and actual CI results are recorded below.
Source-version-label mismatch documented; no silent requirement omission. Dataset license
details and legacy HF script compatibility require Phase 5 resolution. Hardware, compute
budget, training configs, artifact host, deployed service accounts and human submission
actions are future decisions, not Phase 1 failures.

Next exact scope: Phase 2 manual three-token attention derivation (dk=dv=2), explicit
scaled dot-product attention, optional causal mask, math/shape/row-sum/future-invariance/
finite-gradient tests, F01/F02, and sequential-vs-parallel computation explanation.
No multi-head engine, positions, full model, data/training, ablation or deployment in Phase 2.
**STOPPED - WAITING FOR EXPLICIT PHASE 2 APPROVAL** after Phase 1 publication.

## GitHub publication and final quality receipt

* `git add .` staged only the 34 inventory files; final precommit index audit PASS,
  largest file 44,429 bytes, zero findings. `git diff --cached --check` passed;
  `ruff check .` passed and `ruff format --check .` reported eight formatted files.
* `git commit -m "Build independent Phase 1 foundation and assignment compliance plan"`
  created `8b7b88c250fccd3045757673521d4415402f00f1` (34 files, 2,046 insertions,
  one original README line replaced).
* `git push -u origin phase-1-foundation` failed inside the network sandbox and
  succeeded on approved outside-sandbox retry. Push created only the feature branch.
* `gh pr create --repo zaidbinnaveed/AI-Engineering-Assignment-01 --base main --head
  phase-1-foundation --title ... --body-file ...` created open reviewable PR:
  https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/pull/1.
  The temporary body file contained the exact review description; no shell-escaped
  multiline body was used. No PR merge occurred.
* `gh pr view 1 ... --json ...`, `gh run list ... --json ...`, `gh run view ... --json ...`
  and `git ls-remote origin refs/heads/main refs/heads/phase-1-foundation` verified PR
  owner/base/head, successful publication and unchanged remote main at source SHA.
* Push CI https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38255520775
  **completed SUCCESS** for Linux CPython **3.11 and 3.14** on implementation commit.
  Each actually installed CPU dependencies/package, passed tests, dependency checks,
  lint/format, config inspection and sdist/wheel build. PR event ran the same jobs;
  its implementation steps also passed. Final documentation receipt is committed
  separately so the original verified implementation commit remains identifiable.
* Source PDF and both instruction-copy hashes remained unchanged. Working-tree
  cleanliness, feature-tip synchronization, final diff/index scan and PR state are
  verified again after the documentation receipt push.

The complete inventory and environment/configuration/next-phase report are in
`phase1_completion_report.md`. **STOPPED - WAITING FOR EXPLICIT PHASE 2 APPROVAL.**

# Phase 2 execution log — 2026-10-12

User explicitly authorized Phase 2 using the newly supplied Desktop instructions.txt.
Re-read it in full, preserved exact bytes in `docs/phase2_instructions.txt`, and kept
historical root `instructions.txt` unchanged. Re-read every listed foundation source,
all 11 PDF pages (text and rendered images), master, architecture, complete matrix,
Phase 1 log/report, config/reproducibility/tests, pyproject and README before code.
The Version 1.1 reference remains a historical discrepancy; supplied master is 1.0.
PDF unchanged SHA256 `30e543d6d429f60bba96f7c20e2131d198551112b018520f6eae6dd8a23c6bdd`.
Temporary PDF renders are ignored under this project's `tmp/pdf-review/`.

## Verified merge prerequisite and branch

`git remote -v`, repo root/status, `gh api user --jq .login`, account permissions,
PR1 head/content/checks were verified. Account zaidbinnaveed; exact authorized origin;
clean worktree. Reviewed head `25c5f627580faa2e144173b6013aac2fb858b053` unchanged;
all four final Phase 1 push/PR Python 3.11/3.14 checks SUCCESS. `gh pr diff --stat`
was unsupported; corrected to `--name-only` plus local Git diffstat.

Executed authorized `gh pr merge 1 --repo zaidbinnaveed/AI-Engineering-Assignment-01
--merge --match-head-commit 25c5f627580faa2e144173b6013aac2fb858b053`.
Rechecked PR1 state MERGED at `2026-10-12T13:38:23Z`, merge commit
`f04dd07cb8c5fad8cd21ae19f111b548283706f8`.
Executed `git fetch origin`, `git switch main`, `git merge --ff-only origin/main`,
`git switch -c phase-2-attention-engine`, `git rev-parse HEAD`, `git status --short`.
Phase 2 branch starts from that exact integrated main commit, not recreated Phase 1.
No Phase 2 merge is authorized or performed.

## Mathematics, code decisions and evidence

Selected the handoff's exactly-three-token example:
Q=K=[[1,0],[0,1],[1,1]], V=[[1,2],[3,0],[0,4]], dk=dv=2.
Derived QKᵀ=[[1,0,1],[0,1,1],[1,1,2]], scaled by sqrt(2). Defined
a=1/sqrt(2), r=exp(a); every dot product, scaled entry, row numerator/denominator,
probability/sum and six unrestricted/causal weighted output components are explicit.
Scalar math calculator provided decimal approximations after exact expressions;
independent Python-loop oracle verifies all seven intermediates and the float64 engine.
Unmasked O≈[[0.994440,2.406673],[1.401112,2],[0.993020,2.510470]].
Causal O≈[[1,2],[2.339523,0.660477],[0.993020,2.510470]].
Maximum engine absolute error 1.1102230246251565e-16; document nine-decimal error
4.845538414954831e-10, inside 5.1e-10 rounding-aware tolerance.

Engine explicitly computes matmul/scaling/masked_fill/row-softmax/matmul. Pure supplied
Q/K/V, no trainable projection/fused attention. Shapes (...,Tq,dk), (...,Tk,dk),
(...,Tk,dv) with identical leading axes; masks alone broadcast to (...,Tq,Tk).
Float32/float64, positive dimensions, finite inputs and common dtype/device. Non-square
unmasked permitted; aligned equal causal lengths required. Boolean True blocks;
additive entries same dtype 0/-inf only. Mask union before softmax; fully blocked rows
raise clearly. No precision conversion, CPU fallback, dropout or projection.

Added complete typed manual derivation and human review notice, independent experiment,
math/shape/mask/numeric/gradient tests, structural mask generator and reasoned
RNN/Transformer comparison. Sources/attribution documented. No RNN benchmark/training.
Pinned Matplotlib 3.10.8 only as development figure dependency. Existing baseline and
all Phase 1 source/tests unchanged. PNG exceptions in index scanner limited to exact
F01/F02 names, <500 KB, PNG signature, artifact/source hashes; all other binary policy
remains. Remote audit accepts the same exact HTTPS repository with/without .git suffix
for actions/checkout compatibility, not another remote.

## Actual verification and corrections

`PY` abbreviates `.\.venv-phase1\Scripts\python.exe`, repository root:

| Command actually executed | Outcome |
| --- | --- |
| `PY -m pytest -q` before Phase 2 changes | Original 54 passed in 66.66s |
| `PY -m pip install matplotlib==3.10.8 --timeout 30 --retries 0` | Sandbox lookup failed; approved network retry installed compatible cp314 Windows wheel |
| `PY -m ruff check . --fix`, `PY -m ruff format .` | Initial compact semicolon style flagged; format corrected, final checks passed |
| Initial attention test runs | 80 passed, 2 failed: expected 1/3 constructed float32 before float64 comparison; corrected dtype, kept strict tolerance |
| `PY -m pytest tests/test_attention.py -q` after correction | 82 passed in 3.61s, zero failed/skipped |
| `PY -m pytest -q` editable | 136 passed in 23.03s, zero failed/skipped; original 54 intact |
| `PY -m pip check` | No broken requirements found |
| `PY -m pip install --no-build-isolation --no-deps -e .` | Editable installation passed |
| `PY -m build --no-isolation` | sdist and wheel built successfully |
| `PY -m pip install --no-deps --force-reinstall dist/zaid_lm-0.1.0-py3-none-any.whl`, then full pytest | Noneditable wheel installed; 136 passed in 186.34s; zero failed/skipped |
| `PY -m experiments.attention_example --figure` | Independent seven-matrix checks PASS; verified F01 and JSON receipts |
| `PY -m experiments.visualize_causal_mask` | Implementation agrees with independent index rule for 3/8 tokens; F02 generated |
| `PY -m pip freeze --exclude-editable` | Recorded exact Phase 2 Windows environment separately |
| `git add .`, `PY scripts/verify_repository.py`, `git diff --cached --check` | 49-file index audit PASS, largest 127,047 bytes, zero findings; staged diff check passed |
| `git check-ignore` env/PDF temp/output PNG/dist wheel | All generated paths ignored; only validated report PNGs tracked |

CPU-thread launch overhead slowed tiny initial gradchecks; a scoped one-thread test
fixture restores the previous setting afterward. This is test scheduling, not model
speed evidence. One edit patch missed formatter-altered context, failed atomically,
then was corrected against actual text. Sandbox GitHub recheck errors resolved with
approved access; authenticated account and merge receipt reverified. Credentials
never printed or written. Local CRLF/LF Git normalization notices were informational;
source SHA256 receipts were checked against the actual staged bytes.

Visually inspected initial and regenerated F01/F02. Initial F02 footer/legend overlapped;
reserved figure margins and re-rendered. F01 row/feature/key labels improved. Final PNGs
fully readable and numerically correct. F01 is 127,047 bytes; F02 is 76,383 bytes;
hashes and generators recorded in neighboring JSON. Ignored `outputs/figures/` retains
generated copies; tracked `report/evidence/` retains only small report-ready copies.

## Scope, human review and scientific limits

Pure attention mathematics, engine, evidence and conceptual explanation are complete.
L01/F02 verified; engineering acceptance table verified. H01-H06/F01/L02/M01 remain
in progress at submission level pending Zaid's personal review; H07 human action.
No fabricated handwritten scan, signature or personal understanding assertion.
I04 projections await Phase 3; I09 full-model causal integration awaits Phase 4.
No GPU verification; no half precision; eager finite-value checks may synchronize
accelerators. No full model, data download/tokenizer, training/checkpoint, benchmark,
ablations, generated model samples, API/deployment/article/submission was started.

Next exact scope, **only after separate approval**: learned multi-head Q/K/V/output
projections, explicit splitting/permutation/concatenation/merging, shape/reference/mask/
gradient tests, genuine F03; own sinusoidal positions, formula/determinism/offset/
dtype/device tests, genuine F04. Later milestones retain their own boundaries.
Phase 2 GitHub publication and final CI receipts are appended only once verified.

## Verified Phase 2 publication and implementation CI

`git commit -m "Implement verified Phase 2 attention, manual derivation and causal evidence"`
created `cfc7db3d9a986d0640b3c97648f88ab24a423877` (24 changed files).
`git push origin phase-2-attention-engine` succeeded with approved network access;
only the dedicated feature branch was created. `gh pr create --repo
zaidbinnaveed/AI-Engineering-Assignment-01 --base main --head phase-2-attention-engine
--title ... --body-file tmp/phase2-pr.md` created
https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/pull/2.
Verified OPEN, exact implementation head and base main
`f04dd07cb8c5fad8cd21ae19f111b548283706f8`. PR2 was not merged.
Temporary PR body is ignored and uses actual newlines, not escaped shell text.

* Push CI https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38260385925
  COMPLETED SUCCESS, Linux CPU Python 3.11 **136 passed in 2.07s**, Python 3.14
  **136 passed in 2.66s**. Read job logs for exact counts, not merely pending checks.
* PR CI https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38260453436
  COMPLETED SUCCESS on both Python jobs at the same implementation head.
* Actual steps confirmed CPU installation, pip check, pytest, lint/format, staged
  evidence/source audit, config inspection, seven-matrix/F01 verification, F02
  generation and package build. Display name initially retained the old workflow
  label, but the new step list and successes were explicitly checked.
* Restored editable development install after the successful noneditable-wheel suite;
  lint passed, all 13 Python files formatted. Full Phase 1 functionality intact.
* Regenerated both figures from unchanged sources: exact same local PNG SHA256 values,
  unchanged tracked receipts and clean worktree. Final audit PASS: 49 tracked files,
  largest 127,047 bytes, no findings. `git diff --cached --check` passed before commit.
* Final report/log receipt is committed separately so verified implementation remains
  identifiable. Final-head CI and open PR state are verified again before delivery;
  final delivery message supplies that exact head SHA.

**Phase 2 engineering COMPLETE. H07/M01 personal mathematics review pending.
STOPPED — WAITING FOR REVIEW AND EXPLICIT PHASE 3 AUTHORIZATION.**

## Phase 3 — explicit multi-head attention and sinusoidal positions (2026-10-12)

Re-read the current Desktop instructions in full and all specified predecessor
files/master/PDF before choosing architecture. The original PDF is unchanged;
11-page text and cached relevant page renders were inspected. Current handoff copied
byte-for-byte to `docs/phase3_instructions.txt`, SHA256
`e958c332c59ddc7ddfea8906d2c9514eee70839feace5f8aa9cc26126c9bc217`.

Gate: clean worktree; exact authorized origin; actual GitHub account `zaidbinnaveed`
with appropriate permissions; remote refs fetched; PR2 OPEN/MERGEABLE at reviewed
`bf9df03a55c16b7dc0018cd4d268c35900f4b336`, all four CI checks SUCCESS. Requested
the separately mandated merge authorization. User explicitly replied **"Authorize
the merge and complete Phase 3"**. Rechecked head/checks, merged via
`gh pr merge 2 --merge --match-head-commit bf9df03...`, verified MERGED at
`2026-10-12T14:26:59Z`. Fetched, fast-forward synchronized main and branched
`phase-3-multihead-position` from `3b8bcd3df0c0d456694b2e6deaeaa09b6ab7b0cd`.
Phase 2 source/tests/config/manual/evidence comparison against reviewed tree was empty.

Implemented independent Q/K/V/output `nn.Linear` modules. Project `(B,T,D)`, reshape
`(B,T,H,Dh)`, permute `(B,H,T,Dh)`, reuse the exact Phase 2 engine at sqrt(Dh),
permute/contiguous-reshape back and project. Output-only dropout leaves returned
normalized weights unchanged. Both dtype/device and invalid dimensions/masks are
checked. Baseline D128/H4/Dh32 has 66,048 biased parameters; bias-free count 65,536.
Own sin/cos formulas compute a persistent positional buffer, support even/odd widths,
add without mutation and accept bounded integer offsets; addition dropout is explicit.
Demonstration composition stops at positions followed by attention, with no decoder.

Independent oracle uses Python-loop projections, feature slices, existing scalar
attention, explicit concatenation and Python-loop output projection. Canonical
float64 asymmetric example error: unmasked output/weights 2.220446049250313e-16;
causal output 8.881784197001252e-16, weights 1.1102230246251565e-16;
rtol=atol=1e-12. Baseline final prefix future-perturbation error 0.0, all future
head probabilities exactly zero. All four projection weight norms are finite/nonzero;
tests check input/all weights/biases and input gradcheck. PE's full 128x128 scalar
reference error: float32 2.979937197000737e-08 at 1e-6 absolute tolerance; float64
4.440892098500626e-16 at 1e-12. Position zero, pair identity, offsets, odd widths,
state roundtrip, dtype/device migration and qualified permutation explanation verified.

Actual local commands use `.venv-phase1/Scripts/python.exe`:

* `-m pytest tests/test_config.py tests/test_foundation.py tests/test_attention.py -q`:
  all original **136 passed in 98.54s** before coding. Older tests unchanged.
* `-m pytest tests/test_multihead_attention.py tests/test_positional_encoding.py
  tests/test_phase3_integration.py -q`: **105 passed in 44.33s**.
* `-m experiments.multihead_shape_trace`: PASS; captured actual projections/merged
  projection input, separately inspected head/scores, compared real forward outputs,
  scalar references/parameter/causal/gradient checks. Text F03 saved to ignored
  outputs/logs and reviewed tracked copy with content/source receipt.
* `-m experiments.visualize_positions`: PASS; implementation values independently
  verified then heatmap/curves F04 generated. Visually reviewed axes, title, scale
  and formula. PNG is 313,423 bytes; receipt hashes sources and exact artifact bytes.
* `-m pip install --no-build-isolation --no-deps -e .`, `-m pip check`: editable
  install succeeds, no broken requirements. Existing dependency pins unchanged.
* `-m build --no-isolation`: sdist and wheel built. Installed wheel with
  `-m pip install --no-deps --force-reinstall dist/zaid_lm-0.1.0-py3-none-any.whl`;
  import path asserted inside actual site-packages, including both new modules.
* `-m pytest -q` against that noneditable wheel: **241 passed in 16.04s**,
  zero failures/skips, including original 136 and all new 105 tests.
* Ruff lint PASS; formatting check PASS, all 21 Python files formatted.
* `git diff --check` PASS. Exact supplied instruction hashes match.

Resolved issues: reran formatter after a long example-bias line; construct the fixed
example biases directly in float64 to avoid previously rounded float32 literals.
No scientific test failed. Documentation distinguishes fixed/untrained evidence,
causal-mask ordering, `.double()` casting versus initial precision, and CPU-only checks.
No old engine, mathematical derivation, reference source, F01/F02 or config was changed.
Full inventory, artifact hashes, risks and exact remaining scope are in
`docs/phase3_completion_report.md`. Provenance/security audit and final publication/CI
receipts will be appended after actual execution; no pending check is called successful.

Staged audit PASS: 65 tracked files, largest 313,423 bytes, zero findings; all four
evidence artifacts and source receipts verified. Source instruction hard-break spaces
initially triggered the newly staged diff check; preserved those exact bytes with a
path-specific whitespace attribute. F03 text uses explicit LF checkout to preserve
its artifact hash across platforms. Other implementation whitespace checks stay active.

Phase 3 verifies L03/L04/M02/I03/I04/I05/F03/F04. H07/M01 personal arithmetic
review remains pending; I09 full decoder integration, F05-F10 and later requirements
remain incomplete. Tensor-value checks impose unprofiled overhead; no GPU, corpus
download, model training, KV cache, API or article publication occurred. Next prerequisites:
review Phase 3 PR, explicit merge authorization, explicit Phase 4 approval. Planned
Phase 4 is decoder/GPT architecture plus controlled MLM/CLM comparison only.

### Phase 3 publication and CI receipt

Reviewed exactly 24 legitimate Phase 3 changed files; `git diff --cached --check`
PASS after preserving supplied hard-break spaces. Commit
`36d35c44fcabc718231f8b7c3795cbb4e73c7206` contains the verified implementation.
Pushed only `phase-3-multihead-position` to the authorized origin. Opened
https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/pull/3 against main using
`gh pr create --repo zaidbinnaveed/AI-Engineering-Assignment-01 --base main --head
phase-3-multihead-position --title ... --body-file tmp/phase3-pr.md`. PR body uses
actual newlines and is ignored. PR3 remains OPEN/MERGEABLE, not merged.

Push CI https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38266041254
COMPLETED SUCCESS: Python 3.11 editable **241 passed in 2.35s**, wheel **241 passed
in 1.89s**; Python 3.14 editable **241 passed in 3.62s**, wheel **241 passed in
2.31s**. Read actual job logs to verify counts, pip check, lint, 21-file format check,
Git-index audit, config, all F01-F04 generators and sdist/wheel build.
PR CI https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01/actions/runs/38266150363
COMPLETED SUCCESS on both versions at the same implementation head. All four
checks SUCCESS; GitHub display retained old workflow label, actual new steps checked.

Restored editable development install. Regenerated F03/F04 from committed sources:
identical local content hashes and receipts, clean worktree. Documentation audit
checks all new local links, exact inventory, supported new VERIFIED requirement IDs
and unchanged old code/tests/manual/config. Its initial temporary regex also matched
command-table rows; scoped it to the inventory section before final verification.
This was an audit-script selection issue, not a source or numerical test failure.

Final completion report and log are committed separately to preserve the identifiable
verified implementation head. Final documentation-head CI and OPEN PR state are
verified before delivery; that exact delivery SHA/results are in the final message.
**Phase 3 engineering COMPLETE. H07/M01 personal mathematics review pending.
STOPPED — WAITING FOR INDEPENDENT REVIEW AND EXPLICIT PHASE 4 APPROVAL.**
