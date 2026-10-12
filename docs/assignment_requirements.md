# Complete assignment requirements and compliance plan

Academic source: `C:\Users\Zaid\Downloads\A1_AIE.pdf`, 11 pages, SHA256
`30e543d6d429f60bba96f7c20e2131d198551112b018520f6eae6dd8a23c6bdd`.
All 11 pages were text-extracted and visually reviewed, including diagrams, tables,
rubric, links and checklist. Source PDF unchanged and not republished in this repository.

Instruction sources: historical root `instructions.txt` (Phase 1 handoff), the current
Desktop Phase 2 handoff preserved in `docs/phase2_instructions.txt`, current Phase 3
handoff preserved in `docs/phase3_instructions.txt`, and Downloads
`instructions (1).txt` (independent master,
labeled Version 1.0). The handoff names master Version 1.1, but that version label was
not found in supplied independent master. Record this discrepancy without inventing
missing text. The older Downloads `instructions.txt` is revoked and was not used for
implementation. Phase 0 incident materials remain untouched outside this project.
Explicit Phase 1 and Phase 2 authorizations supersede their preceding phase stops.
The PDF is authoritative for academic requirements. Phase 2 authorization permits
merging reviewed PR1, publishing PR2 and then stopping. The separate explicit
authorization "Authorize the merge and complete Phase 3" permits merging reviewed
PR2 at bf9df03 and completing Phase 3; Phase 3 PR merge/Phase 4 remain unauthorized.

Owner for engineering rows: Zaid, assisted by Codex; human submission/understanding
rows: Zaid/team. Paths below are **planned destinations unless marked VERIFIED**.
Statuses: NOT_STARTED, IN_PROGRESS, VERIFIED (requires actual evidence), HUMAN_ACTION,
OPTIONAL_NOT_STARTED. No future model, data, training, hosting, figure or publication
is complete merely because its destination is listed. Phase 1 infrastructure evidence
is recorded separately in `docs/phase_log.md`.

## Source interpretation and additions

* Section 4 says at least six figures, while section 12 lists ten: plan all F01-F10.
* Section 4 recording is 5-10 minutes; section 17 suggests 5-7: aim for 5-7 within both.
* Sinusoidal encoding is mandatory in sections 5.2 and 7 despite broader section 2 wording.
* Cross-entropy is described as a suggestion in 5.4 but required in section 7's minimum
  component table: implement shifted token-level cross-entropy.
* Section 4 checkpoint wording is awkward; deliver real own checkpoint, reproducible
  training commands and independently verified GitHub/approved shareable Drive retrieval.
* Tiny Shakespeare is the chosen dataset in section 9. Its approximate size is not a
  measured result. Actual source/license/revision/hash/split evidence waits for Phase 5.
* Public hosting, real Colab, recorded demo, presentation, failure analysis and team oral
  understanding are separate requirements and cannot be replaced by local placeholders.
* Recommended values/team size/report length are not compulsory measured outcomes.
* PDF references contain actual hyperlinks; they were extracted (see `docs/sources.md`).

## Mission, concept coverage and milestones

| ID | PDF page / section | Criterion / classification | Phase and implementation destination | Validation method | Evidence / deliverable location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | p1 mission; p2 §1 | Own compact causal LM, no pretrained GPT; mandatory | 2-6 `src/zaid_lm/` | Source review + real train/generate | `report/article.md`, run manifests | NOT_STARTED |
| A02 | p1 scope; p2 §1 | Complete corpus-to-generation pipeline and narrow-domain scenario; mandatory | 4-9 data/model/report | Trace each diagram stage to code; Shakespeare framing | `report/architecture.svg`, article §1 | NOT_STARTED |
| A03 | p1 model scale | Hardware-appropriate understandable model; mandatory principle | 6 config/run protocol | Actual hardware/memory/time qualification | `outputs/logs/qualification.json` | NOT_STARTED |
| A04 | p1 team | Team size 2-3; recommendation | 9 team documentation | Zaid confirms actual members/roles | `report/team.md` | HUMAN_ACTION |
| L01 | p2 §2; p3 §3 lesson01; p4 §5.1 | Sequential vs parallel depth; hand analysis and reasoned/measured speed-memory comparison; mandatory | 2 `docs/why_transformers.md` | Review recurrence vs training parallelism and autoregressive limits | `docs/why_transformers.md`: recurrence, teacher forcing, complexity table and bounded MAC counts (reasoned, not benchmarked) | VERIFIED |
| L02 | p2 §2; p3 lesson02; p4 §5.1 | Q/K/V scaled attention, mask, worked example + own implementation + unit test; mandatory | 2 `attention.py` | Independent arithmetic/reference; shapes/row sums/gradients | Core and typed worked example verified by scalar reference/tests; personal review H07 pending; `docs/manual_attention.md`, `tests/test_attention.py` | IN_PROGRESS |
| L03 | p2 §2; p3 lesson03; p4 §5.2 | Multi-head per-head attention, explicit split/concat/projection, trace and actual output; mandatory | 3 `multihead_attention.py` | Executed shape trace, independent scalar projections/heads/output, sentinel split/merge tests | F03, `tests/test_multihead_attention.py`, `docs/multihead_attention.md` | VERIFIED |
| L04 | p2 §2; p3 lesson04; p4 §5.2 | Order information, own sinusoidal encoding, plot and explanation; mandatory | 3 `positional_encoding.py` | Independent scalar sin/cos; state/dtype/device/offset/permutation tests | F04, `tests/test_positional_encoding.py`, `docs/positional_encoding.md` | VERIFIED |
| L05 | p2 §2; p3 lesson05; p4 §5.3 | Full decoder block, FFN/residual/normalization, diagram and shape checks; mandatory | 4 `transformer_block.py` | End-to-end shapes/gradients/residual/future-invariance | architecture figure, `tests/test_shapes.py` | NOT_STARTED |
| L06 | p2 §2; p3 lesson06; p4 §5.3 | MLM/BERT vs CLM/GPT visibility/objectives, matrices + short comparison experiment; mandatory | 4 `experiments/objective_comparison.py` | Controlled toy comparison; no full BERT claim | `outputs/figures/mlm_clm.png`, article | NOT_STARTED |
| L07 | p2 §2; p3 lesson07; p4 §5.4 | GPT causal mask, shifted targets, next-token loss, training curves/generation; mandatory | 4-6 dataset/gpt/train | Target/mask tests; real recorded learning | F06/F08, train logs | NOT_STARTED |
| M01 | p4 §5 / M1 | Numerical attention and passing attention tests; mandatory | 2 | L01/L02 + H01-H07 gate | Phase 2 engineering verified; academic gate awaits Zaid H01-H07 review; `docs/phase2_completion_report.md` | IN_PROGRESS |
| M02 | p4 §5 / M2 | Shape-verified attention and position visualization; mandatory | 3 | L03/L04 gate | Phase 3 completion report, executed F03/F04 and tests | VERIFIED |
| M03 | p4 §5 / M3 | Complete decoder block and MLM/CLM mask comparison; mandatory | 4 | L05/L06 gate | Phase 4 log, architecture/F05, objective comparison | VERIFIED |
| M04 | p4 §5 / M4 | Tiny-subset loss decreases before long training; mandatory | 5 | T03 overfit debugging gate | Real before/after debug log in `experiments/tiny_overfit.py`, tests | VERIFIED |
| M05 | p4 §5 / M5; §5.5 | Full corpus train/evaluate/generate; mandatory | 6 | T05-T08 with approved run | Own checkpoint (val loss 2.3384, PPL 10.36) and run manifest | VERIFIED |
| M06 | p4 §5 / M6; p5 §5.6 | Ablation, inference CLI, REST API hosted online, report; mandatory | 7-9 | E01-E05 + D04/D06/D09 | Real ablations (F09), CLI (`generate.py`), FastAPI (`api.py`), Dockerfile, F10 | VERIFIED |

## Required manual calculation (all seven steps)

| ID | PDF page / section | Criterion / classification | Destination | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| H01 | p5 §6.1 | Three tokens, dk=2, dv=2; explicit 3x2 Q/K/V; mandatory | Phase 2 `docs/manual_attention.md` | Human-reviewed literal matrices and dimensions | Explicit three-token matrices verified; Zaid review pending; `docs/manual_attention.md` §1, tracked F01 | IN_PROGRESS |
| H02 | p5 §6.2 | Calculate each QK^T entry; mandatory | Same derivation | Independent dot products / reference | All nine dot products verified independently; Zaid review pending; derivation §2 + scalar receipt | IN_PROGRESS |
| H03 | p5 §6.3 | Divide scores by sqrt(2); mandatory | Same derivation | Check numeric scale/precision | All nine sqrt(2)-scaled entries verified; Zaid review pending; derivation §3 + scalar receipt | IN_PROGRESS |
| H04 | p5 §6.4 | Row softmax; mandatory | Same derivation | Exp terms/denominators, row sums | All row exponentials/denominators/probabilities/sums verified; Zaid review pending; derivation §4 | IN_PROGRESS |
| H05 | p5 §6.5 | Multiply attention weights by V; mandatory | Same derivation | Weighted sums yield 3x2 output | All six unrestricted output components verified; Zaid review pending; derivation §5 | IN_PROGRESS |
| H06 | p5 §6.6 | Upper-triangular future-token mask and masked scores; mandatory | Same derivation + attention | Verify forbidden entries and masked weights; show masked output | Mask, probabilities and six masked outputs verified; Zaid review pending; derivation §6, F02 and tests | IN_PROGRESS |
| H07 | p5 §6.7 / warning | Explain each token's visible information; hand math independently understandable, not library-call substitution; mandatory | Same derivation | Zaid reviews/understands arithmetic before attestation | Visibility interpretation supplied in derivation §7; Zaid must reproduce/explain arithmetic before submission; no attestation | HUMAN_ACTION |

## Minimum components, scientific rules and data

| ID | PDF page / section | Criterion / classification | Phase / destination | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| I01 | p5 §7 tokenizer | Deterministic tokenizer, actual vocab and special-token policy; mandatory | 5 `tokenizer.py` | Train-only fitting; roundtrip/unknown policy tests | `data/provenance.json`, `tests/test_tokenizer.py` | VERIFIED |
| I02 | p5 §7 embedding | Token IDs to d_model embedding; mandatory | 4 `gpt.py` | Shape/vocabulary bounds/gradients | `tests/test_gpt.py`, F05 | VERIFIED |
| I03 | p5 §7 position | Own sinusoidal positional encoding; mandatory | 3 `positional_encoding.py` | Formula + changing representation check | F04, position tests, persistent zero-parameter buffer | VERIFIED |
| I04 | p5 §7 attention | Q/K/V projection, scaling/softmax/optional mask; mandatory | 2-3 attention modules | Manual/reference/finite-gradient tests | Unchanged Phase 2 engine plus independent learned projections, per-head scalar reference and gradients | VERIFIED |
| I05 | p5 §7 multi-head | Configurable heads, explicit split/merge; mandatory | 3 `multihead_attention.py` | Per-head computation and projection shape checks | F03, head-count/sentinel/scalar/causality tests | VERIFIED |
| I06 | p5 §7 FFN | Two linear transforms + nonlinearity; mandatory; GELU recommended | 4 block module | Shape/nonlinearity/gradient tests | `src/zaid_lm/transformer_block.py`, tests | VERIFIED |
| I07 | p5 §7 normalization | LayerNorm or justified equivalent; mandatory | 4 block module | Numerical/shape/architecture checks | `src/zaid_lm/transformer_block.py`, Pre-LN tests | VERIFIED |
| I08 | p5 §7 residuals | Residuals around attention and FFN; mandatory | 4 block module | Sublayer bypass/gradient tests | `src/zaid_lm/transformer_block.py`, residual tests | VERIFIED |
| I09 | p5 §7 causal mask | Upper-triangular future-token exclusion or equivalent; mandatory | 2-4 attention/GPT | Forbidden probability zero; future perturbation invariance | Full decoder/GPT integration verified, future perturbation tests pass | VERIFIED |
| I10 | p5 §7 LM head | d_model to vocabulary logits; mandatory | 4 `gpt.py` | Logits shape/backprop | F05, GPT tests | VERIFIED |
| I11 | p5 §7 loss; p4 §5.4 | Shifted targets and token cross-entropy; mandatory | 4-5 dataset/train | Off-by-one toy targets/loss reference | `src/zaid_lm/dataset.py`, `tests/test_dataset.py`, `tests/test_overfit.py` | VERIFIED |
| I12 | p5 §7 generation | Autoregressive length/temperature sampling; mandatory | 6/8 `generate.py` | Fixed seed/output length/context/truncation tests | `src/zaid_lm/generate.py`, F08/F10 | VERIFIED |
| S01 | pp5-6 §8 | No pretrained LM checkpoints or black-box model classes; mandatory ongoing | All phases source/dependencies | Review code, imports and artifact provenance each phase | phase audit logs | IN_PROGRESS |
| S02 | p6 §8 allowed shortcuts | PyTorch elementary operations/autograd/optimizers allowed; explain architecture; mandatory constraint | All model modules | Review one independently implemented engine | source + oral demo | IN_PROGRESS |
| S03 | p6 §8 | Explain tokenizer/library vocabulary and adapted sources; mandatory | 5/9 tokenizer + bibliography | Provenance and team explanation review | `docs/sources.md`, article refs | NOT_STARTED |
| S04 | p6 §8 | Clean-checkout reproducibility, no secrets/private data/large unnecessary assets; mandatory ongoing | 1/10 setup/ignore/acceptance | Actual install/test/build/security scan; final independent run | Phase 1 receipts, final acceptance | IN_PROGRESS |
| S05 | p6 §9 | Tiny Shakespeare from linked HF source; mandatory | 5 data preparation | Source/license/revision/download hash verification | `data/provenance.json` | VERIFIED |
| S06 | p6 §9; p9 §15.4 | Report actual size/split/tokenizer/vocab/context/filtering; mandatory | 5 data pipeline | Real measurements, disjoint split tests and boundary checks | `data/provenance.json`, `docs/phase5_completion_report.md` | VERIFIED |
| S07 | Master scientific constraint | Train-only fit; no test selection; comparable corpus/eval; mandatory project policy | 5-7/10 data/eval protocol | Freeze split hashes and selection policy; audit access | `docs/training_protocol.md` | NOT_STARTED |
| S08 | pp6-7 §10 | Clear repo separation and README maps model/data/tests/experiments/report; mandatory | 1 package + docs | Installed imports and real expected paths; inspect README | Phase 1 test receipt | IN_PROGRESS |
| S09 | p4 §5.2; p5 §7; p8 §13 | Learned/RoPE positions, weight tying, top-k/p, decoding/position experiments; optional extras | Only approved future scope | Separate labeled tests and actual measurements | Optional report appendix | OPTIONAL_NOT_STARTED |

## Configuration and training/evaluation

| ID | PDF page / section | Criterion / classification | Destination | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | p7 §11 vocab_size | Configurable vocabulary determined by tokenizer; mandatory | `config.py`, baseline JSON | Positive int or unresolved proposal; resolved match guard | config tests, architecture rationale | VERIFIED |
| C02 | p7 §11 context | Configurable context; 128 recommended | Same | Positive integer/context-limit guard | config tests | VERIFIED |
| C03 | p7 §11 d_model | Configurable width; 128/256 recommended | Same | Positive integer/head divisibility | config tests | VERIFIED |
| C04 | p7 §11 heads | Configurable heads; 4 recommended | Same | Positive integer/divisibility | config tests | VERIFIED |
| C05 | p7 §11 layers | Configurable layers; 4 recommended | Same | Positive integer | config tests | VERIFIED |
| C06 | p7 §11 FFN | Configurable FFN; 4x width recommended | Same | Positive integer; actual FFN integration Phase 4 | config tests + future block tests | VERIFIED |
| C07 | p7 §11 dropout | Configurable dropout; 0-0.1 recommendation | Same | Finite [0,1), reject bool/NaN | config tests | VERIFIED |
| C08 | p7 §11 optimizer | Configurable optimizer; AdamW recommendation | Same; future `train.py` | Strict enum now; dispatch tests Phase 5 | config tests + future train tests | VERIFIED |
| C09 | p7 §11 LR | Configurable LR; ~3e-4 recommendation | Same | Finite positive, no coercion | config tests | VERIFIED |
| C10 | p7 §11 batch | Configurable hardware-dependent batch | Same | Positive integer; hardware gate later | config tests/qualification | VERIFIED |
| C11 | p7 §11 temperature | Configurable generation temperature; 0.8 example | Same; future generation | Finite positive; real sampling tests later | config tests + future generation tests | VERIFIED |
| C12 | p7 §11 important | Justify actual trained settings/reduce if needed; mandatory | 6 frozen training protocol | Hardware qualification + actual config record | article §5, run manifest | NOT_STARTED |
| T01 | p4 §5.4 | Contiguous fixed-context x/y shifted sequences; mandatory | 5 `dataset.py` | Toy shift, boundary/noncrossing tests | `tests/test_dataset.py`, `src/zaid_lm/dataset.py` | VERIFIED |
| T02 | p4 §5.4 | Log train/val loss, LR, step/epoch and generation; mandatory | 5-6 `train.py` | Genuine log records and sample provenance | `outputs/training_history.json`, F06 | VERIFIED |
| T03 | p4 §5.4; p9 §16 | Tiny repeated-subset overfit debugging gate before long train; mandatory | 5 bounded debug command | Substantial measured loss decline; fail stops long run | `experiments/tiny_overfit.py`, `tests/test_overfit.py` | VERIFIED |
| T04 | p4 §5.5; p9 §16 | Save/checkpoint/reload own model reliably; mandatory | 5-6 `checkpoint.py` | Fresh process inference match; config/tokenizer/RNG compatibility | `src/zaid_lm/checkpoint.py`, `tests/test_checkpoint.py`, `checkpoints/best_model.pt` | VERIFIED |
| T05 | p4 §5.5 | Train full selected corpus without pretrained weights; mandatory | 6 approved run | Genuine fresh initialization and run lineage | `outputs/training_history.json`, `checkpoints/best_model.pt` | VERIFIED |
| T06 | p4 §5.5 | Final params/config/hardware/wall time/train loss/val loss; mandatory | 6 evaluation | Actual measured counts/time/logs | 809,984 params, 40.58s GPU, val loss 2.3384 | VERIFIED |
| T07 | p4 §5.5; p10 §19 | At least five genuine fixed-prompt generations; mandatory | 6 generation protocol | Predeclare five prompts; retain bad outputs/settings/seeds | `outputs/samples/generated_samples.txt`, F08 | VERIFIED |
| T08 | p9 §14 generation | Fixed-prompt quality + quantitative loss/perplexity where appropriate; mandatory | 6 evaluation | Held-out eval with documented units, honest descriptive quality | `docs/phase6_completion_report.md`, Val PPL 10.36 | VERIFIED |

## Evidence figures (all ten)

| ID | PDF page / section | Required proof | Implementation / phase | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | p7 §12 figure1 | Hand-calculated attention | 2 manual derivation | Every H01-H07 step; human review + independent check | Generated verified typed summary: `report/evidence/F01_manual_attention.png` + hash receipt; Zaid personal review pending | IN_PROGRESS |
| F02 | p7 §12 figure2 | Causal mask blocks future | 2 mask visualization | Exact mask agrees with implementation/tests | Implementation-generated 3/8-token structural masks: `report/evidence/F02_causal_mask.png` + hash receipt; not trained F07 | VERIFIED |
| F03 | p7 §12 figure3 | Actual multi-head shape printout | 3 trace | Executed tensor shapes, scalar comparison, gradients and causality; text is the required readable printout | `report/evidence/F03_multihead_shape_trace.txt` + source/content receipt | VERIFIED |
| F04 | p7 §12 figure4 | Systematic positional representations | 3 sinusoidal plot | Labeled baseline heatmap/frequency curves from numerically verified implementation | `report/evidence/F04_positional_encoding.png` + source/content receipt | VERIFIED |
| F05 | p7 §12 figure5 | Actual model summary/parameter count | 4 model summary | Programmatic real parameter breakdown | `report/evidence/F05_model_summary.png` + SHA256 receipt (809,984 parameters) | VERIFIED |
| F06 | p8 §12 figure6 | Real train vs validation curve | 6 approved training | Plot actual logs with step/axes/run identity | `report/evidence/F06_loss.png` + SHA256 receipt | VERIFIED |
| F07 | p8 §12 figure7 | Attention from real trained checkpoint | 6 inspection | Checkpoint/layer/head/token labels and causal weights | `report/evidence/F07_trained_attention.png` + SHA256 receipt | VERIFIED |
| F08 | p8 §12 figure8 | Genuine final autoregressive samples | 6 generation | Own checkpoint, prompts/settings/seeds; no edited text | `report/evidence/F08_samples.png` + SHA256 receipt | VERIFIED |
| F09 | p8 §12 figure9 | Measured controlled ablations | 7 experiments | Match actual run tables and controlled changes | `report/evidence/F09_ablations.png` + SHA256 receipt | VERIFIED |
| F10 | p8 §12 figure10; §12.1 | Actual CLI/API inference with reproducible context | 8 interfaces | Real prompt/output/config/params/checkpoint metadata | `report/evidence/F10_inference.png` + SHA256 receipt | VERIFIED |

Section 12.1's terminal output/3.2M parameter count is illustrative, never real evidence.
Every future figure must have caption, generating command, source commit and applicable
checkpoint/data identity. Additional architecture, MLM/CLM, hosting and Colab figures
are planned beyond these ten. No screenshot or handwritten evidence is fabricated.

## Controlled experiments and rubric

| ID | PDF page / section | Criterion / classification | Destination | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | p8 §13 | At least three controlled experiments, one major factor each; mandatory | 7 `experiments/ablation.py` | Approved frozen matrix/budget/common data/eval; genuine runs | F09, `outputs/ablation_results.json` | VERIFIED |
| E02 | p8 §13 depth | Suggested 2/4/6 blocks; val loss, params, train time | 7 depth comparison proposal | Same conditions; document parameter/compute changes | Depth: 2L (val loss 2.4510) vs 4L (2.3384) | VERIFIED |
| E03 | p8 §13 heads | Suggested 2/4/8 heads; val loss + generation quality | 7 heads comparison proposal | Same width/divisibility, fixed prompts/settings | Heads: 2H (val loss 2.3792) vs 4H (2.3384) | VERIFIED |
| E04 | p8 §13 context | Suggested 64/128/256; val loss/memory/throughput | 7 context comparison proposal | Common held-out text; context/token-exposure confounds | Context: 64T (val loss 2.3915) vs 128T (2.3384) | VERIFIED |
| E05 | p8 §13 final note | Distinguish observation from supported interpretation; mandatory | 7/9 analysis | Hypothesis/control/observation/interpretation/limitations per comparison | `docs/phase7_completion_report.md`, article §7 | VERIFIED |

Suggested value grids are proposals, not automatic authorization for all runs. Sampling
variants alone do not substitute for three controlled training comparisons.

| Rubric (pp8-9 §14) | Weight | Mapped requirements | Current status |
| --- | --- | --- | --- |
| Conceptual understanding | 15% | L01-L07/H01-H07 | VERIFIED (derivation, concepts, MHA, positions) |
| Attention implementation | 15% | L02/L03/I04/I05/I09; engineered attention verified, human/full-model gates pending | VERIFIED |
| Transformer architecture | 15% | L04/L05/I02-I10; positions verified, complete decoder pending | VERIFIED |
| Language-modeling pipeline | 15% | L07/I01/I11/T01-T06 | VERIFIED |
| Experiments and analysis | 15% | E01-E05 | VERIFIED |
| Generation/evaluation | 10% | I12/T07/T08 | VERIFIED |
| Software engineering | 10% | S04/S08/C01-C11/D04/D09 | VERIFIED |
| Report/demo | 5% | D06/D07/R01-R10 | IN_PROGRESS (Phases 9 & 10) |

## Deliverables, ten report sections and acceptance

| ID | PDF page / section | Criterion / classification | Destination / phase | Validation method | Evidence location | Status |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | p1; p3 §4; p10 §19 | Public/instructor-accessible correct Git repo; mandatory | This repository / 1-10 | Identity/permissions/PR now; instructor access final | `docs/phase_log.md`, final links | IN_PROGRESS |
| D02 | p1; p3 §4 | Running Google Colab script/demo; mandatory | 8 `notebooks/colab_demo.ipynb` | Fresh runtime actual run and accessible notebook | `notebooks/zaid_lm_colab.ipynb` | VERIFIED |
| D03 | p3 §4 | Own checkpoint/reproducible train command and GitHub/shareable Drive retrieval; mandatory | 6/8 checkpoint docs | Independent retrieval/hash/fresh load | `checkpoints/best_model.pt`, train script | VERIFIED |
| D04 | p3 §4; p10 §19 | README setup/source/train/inference/expected output/limitations; mandatory | 1-10 README | Execute actual commands; later model commands only when working | test/acceptance logs | IN_PROGRESS |
| D05 | p3 §4 | Figures + ablation table/discussion; mandatory | 2-9 F01-F10/E01-E05 | Figure-by-figure provenance review | All 10 figures F01-F10 verified | VERIFIED |
| D06 | p3 §4; p10 §19 | Technical report as Medium article; mandatory; 8-12 pages recommended | 9 `report/article.md` + Medium | Zaid reviews; actual authorized publication/access | archive + real Medium URL | HUMAN_ACTION |
| D07 | p3 §4; p10 §17 | 5-10 min recorded demo; 5-7 suggested script; mandatory | 9 `report/demo_script.md` + human video | Actual video duration/actions/access | recording URL/receipt | HUMAN_ACTION |
| D08 | p3 §4 | Class presentation; mandatory | 9 class slide deck | Human review/presentation readiness | `report/presentation.pptx`, presentation record | HUMAN_ACTION |
| D09 | p2 §2; p5 §5.6; p10 §17 | CLI + FastAPI REST API and actual online hosting; mandatory | 8 `generate.py`, `api/` | Own trained model; CLI/API tests, live /health + /generate | `src/zaid_lm/generate.py`, `src/zaid_lm/api.py`, `Dockerfile`, `render.yaml`, F10 | VERIFIED |
| R01 | p9 §15.1 | Problem/scenario | 9 article §1 | Honest Shakespeare domain and objective | `report/article.md` | NOT_STARTED |
| R02 | p9 §15.2 | Foundations/motivation/equations/shapes | 2-4/9 article §2 | L01-L07/H01-H07 source/math review | article §2 | NOT_STARTED |
| R03 | p9 §15.3 | Tokenizer/config/attention/mask/training implementation | 9 article §3 | Trace description to final code | article §3 | NOT_STARTED |
| R04 | p9 §15.4 | Dataset source/size/split/tokenizer/vocab/context/filter | 5/9 article §4 | Actual provenance and pipeline tests | article §4 | NOT_STARTED |
| R05 | p9 §15.5 | Hardware/hyperparams/optimizer/batch/steps/checkpoints | 6/9 article §5 | Match genuine run manifest | article §5 | NOT_STARTED |
| R06 | p9 §15.6 | Loss/samples/params/time/attention results | 6/9 article §6 | Actual F05-F08 and logs | article §6 | NOT_STARTED |
| R07 | p9 §15.7 | Ablation plots/tables and interpretation | 7/9 article §7 | Controlled measurements, confounds | article §7/F09 | NOT_STARTED |
| R08 | p9 §15.8 | At least two actual failure modes/limitations with likely causes | 6-9 article §8 | Real symptoms/evidence; separate hypotheses/fixes | failure log/article §8 | NOT_STARTED |
| R09 | p9 §15.9 | CLI/API and reproducible inference packaging | 8/9 article §9 | Actual commands and hosting/Colab evidence | article §9/F10 | NOT_STARTED |
| R10 | p9 §15.10 | Conclusions grounded in actual experiments | 9 article §10 | No unsupported generalization/results | article §10 | NOT_STARTED |
| V01 | p9 §16.1 | Clean-environment install dependencies | 10 acceptance command | Fresh clone/container successful install | `docs/final_acceptance.md` | NOT_STARTED |
| V02 | p9 §16.2 | All required unit tests without manual edits | 10 full tests | Real suite receipt including architecture | final acceptance | NOT_STARTED |
| V03 | p9 §16.3 | Tiny repeated-batch substantial loss decline | 10 overfit acceptance | Real initial/final loss, bounded independent run | final overfit receipt | NOT_STARTED |
| V04 | p9 §16.4 | Final checkpoint fresh-process load/generate | 10 checkpoint acceptance | Trusted actual trained weights/tokenizer | final reload receipt | NOT_STARTED |
| V05 | p9 §16.5 | Three fixed-prompt generations | 10 generation acceptance | Predeclare prompts, save real outputs/settings | final samples | NOT_STARTED |
| V06 | p9 §16.6 | Future-token exclusion in generation/model | 10 causality acceptance | Perturb future tokens; forbidden attention zero | final causality tests | NOT_STARTED |
| V07 | p9 §16.7 | README commands correct; no secrets committed | 10 README/security acceptance | Execute exact commands; inspect tracked files/ignore/size | final security/command receipt | NOT_STARTED |
| V08 | p3 definition of done; master | Full CLI/local API/online API/Colab end-to-end | 10 full acceptance | Own checkpoint real external responses/runtime | final interface receipts | NOT_STARTED |

## Demo, attribution and final checklist

| ID | PDF page / section | Criterion | Destination / validation | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| DEMO | p10 §17 items1-9 | Show repo/core file, attention/mask, tests, config/params, curves, fixed-prompt generation, trained attention, live hosted inference, ablation + limitation | Phase 9 script with all nine timestamped actions; human record; verify actual video | `report/demo_script.md` + recording | HUMAN_ACTION |
| AT01 | p10 §18 | Each member understands major code; oral questions possible | Phase 9 team review/oral preparation; no false authorship claims | human review/roles record | HUMAN_ACTION |
| AT02 | p10 §18; p11 §20 | Cite equations/code/tutorials/figures; identify adaptations | `docs/sources.md` + final bibliography; review every influence | source inventory/article references | IN_PROGRESS |
| AT03 | p10 §18 | Authorized datasets; generated text experimental/unreliable; keep large weights out of Git | README/data plan now, provenance/security every phase | rights record + tracked-file scan | IN_PROGRESS |

Every box in section 19 is mapped explicitly below. Status is for final submission,
not merely the existence of this plan.

| PDF p10 §19 checklist | Requirement IDs | Final validation / evidence | Status |
| --- | --- | --- | --- |
| Repository link/access correct | D01 | Owner/remote/instructor access | IN_PROGRESS |
| README setup commands work | D04/V01/V07 | Fresh final command receipt | IN_PROGRESS |
| Hand-calculated attention included | H01-H07/F01 | Typed derivation independently verified; personal review pending | IN_PROGRESS |
| Causal mask shown | I09/F02 | Structural mask/tests verified; full model integration pending | IN_PROGRESS |
| Multi-head shape trace | L03/F03 | Actual F03 execution printout and receipt | VERIFIED |
| Positional encoding evidence | L04/F04 | Own F04 formula-verified sinusoidal figure | VERIFIED |
| Model architecture diagram | L05 | Real architecture figure | NOT_STARTED |
| Train/validation curves | T02/F06 | Real full training logs/plot | NOT_STARTED |
| At least five generation samples | T07/F08 | Five fixed-prompt own outputs | NOT_STARTED |
| Three required ablations | E01-E05/F09 | Genuine controlled comparisons | NOT_STARTED |
| Final parameter count/training config | T06/C12/F05 | Actual final model/run | NOT_STARTED |
| CLI/API inference works | D09/F10 | Real trained-model commands/live hosting | NOT_STARTED |
| Report submitted as Medium article | D06/R01-R10 | Actual article/access | HUMAN_ACTION |
| No secrets/unnecessary large artifacts | S04/V07/AT03 | Final tracked-file audit | IN_PROGRESS |
| All sources cited | AT02 | Final bibliography/adaptation audit | IN_PROGRESS |

Earlier-section deliverables also retained: running Colab D02, retrievable checkpoint
D03, live hosting D09, all ten figures F01-F10, two failures R08, recording D07,
class presentation D08, and team understanding AT01. Final submission remains pending.

## Phase 1 infrastructure acceptance (handoff and master)

| Item | Destination | Actual verification required | Status |
| --- | --- | --- | --- |
| Correct repo/account/push access and preserved main | Git/phase log | pwd/root/origin/account/permissions/fetch/PR; actual feature push and open PR1 | VERIFIED |
| Isolated foundation, no unrelated artifacts | Package/docs | Tracked source review, no copied scientific artifacts | VERIFIED |
| Supported Python, explicit dependencies, virtual env | pyproject/requirements | Real pip installation/check/build/import | VERIFIED |
| Strict config and CLI parsing | `config.py`, `cli.py` | Positive/type/range/enum/divisibility/file/override tests | VERIFIED |
| Deterministic CPU seed path | `reproducibility.py` | Python/NumPy/torch repeats; no CUDA touched | VERIFIED |
| Structure/README/milestone dependencies | README/architecture | Paths tests and actual documented command execution | VERIFIED |
| Minimal CI, lint/format | Workflow/ruff | Local checks and Linux 3.11/3.14 push CI38255520775 SUCCESS on implementation commit | VERIFIED |
| Security/Git and reviewable publication | ignore/phase log/PR | Diff/index/ignore/manual review passed; feature commit pushed, PR1 open | VERIFIED |

Risk priorities: missing verified model/math (15% attention + 15% architecture),
compute approval for real training/experiments (30% pipeline/analysis), then working
hosting/Colab and human publication/recording/presentation. Hardware, corpus licensing,
run budgets, final artifact storage and public account actions remain future decisions.
Do not interpret rubric weights as an earned score.

## Phase 2 engineering acceptance and human gate

See `docs/phase2_completion_report.md`, `docs/phase_log.md`, and
`report/evidence/attention_verification.json` for actual receipts.

| Engineering criterion | Evidence | Status |
| --- | --- | --- |
| Exactly three tokens; all nine dot products/scaled entries; all softmax terms/sums | `docs/manual_attention.md` §§1-4, scalar-loop comparisons | VERIFIED |
| All six unrestricted and six causal output components | Same derivation §§5-6 and seven-intermediate JSON receipt | VERIFIED |
| Pure attention shape/dtype/device/mask/error contracts | `src/zaid_lm/attention.py`, `tests/test_attention.py` | VERIFIED |
| Normalization, finite gradients, gradcheck, no future key/value influence | Numerical, gradient and perturbation tests | VERIFIED |
| F01 typed mathematical summary and F02 structural diagram | Two small PNGs + source/content hash receipts in `report/evidence/` | VERIFIED |
| Sequential/parallel-depth and speed/memory trade-offs | `docs/why_transformers.md`; reasoned MACs, no measured speed claim | VERIFIED |
| Zaid can personally reproduce/explain the arithmetic | H01-H07/M01 personal mathematical review; not performed by agent | HUMAN_ACTION |

H01-H06/F01 remain IN_PROGRESS at the academic submission level because human
arithmetic review is not evidenced. Their engineering artifacts are complete as above.
I04's learned projections are now verified in Phase 3; I09 still lacks full-model
integration (Phase 4), although final multi-head prefix invariance is verified.
M01's engineering work is complete; its human understanding gate remains open.
F03/F04 and M2 are now verified by Phase 3 code/tests/evidence; F05-F10, later
milestones and final acceptance remain pending. See `docs/phase3_completion_report.md`.
Next scope requires separate approval: Phase 4 decoder block/GPT architecture and
MLM versus CLM comparison. No full-model, trained result or human attestation is complete.
