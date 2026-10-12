# Sources and influence inventory

## Supplied assignment and instructions

* `A1_AIE.pdf`, 11 pages, locally supplied in Downloads. SHA256 and all page/section
  mappings are in `assignment_requirements.md`. All pages visually/textually reviewed.
* `instructions.txt` at repository root is an exact copy of the authorized Desktop
  Phase 1 handoff. `master_instructions_v1.0.txt` is an exact copy of the independent
  master from Downloads `instructions (1).txt`. Version mismatch recorded; neither
  copy alters the original. The revoked older master is not imported.
* No source code/configuration/tokenizer/data/checkpoints/reports/results/history from
  another student's assignment were copied or consulted for implementation. Phase 0
  incident checkout remains untouched outside this dedicated repository.

## Phase 1 implementation influences

Code was authored independently from the PDF and user instructions using Python
dataclasses, JSON, argparse, pathlib and elementary seed APIs. No external model source
was copied. Package/dependency metadata was verified through official PyPI and the
PyTorch CPU wheel index. Pin choices are development infrastructure, not experimental
findings. The following sources support the setup/constraints:

* PyTorch 2.10.0 metadata: https://pypi.org/project/torch/2.10.0/
* NumPy 2.4.6 metadata: https://pypi.org/project/numpy/2.4.6/
* Pytest metadata: https://pypi.org/project/pytest/
* Ruff metadata: https://pypi.org/project/ruff/0.15.4/
* PyTorch reproducibility guidance: https://docs.pytorch.org/docs/2.10/notes/randomness.html

Seed determinism is scoped to unchanged software/hardware; no cross-platform bitwise
guarantee is claimed. CPU tests seed the PyTorch default CPU generator directly so
the CPU smoke path does not ask CUDA to execute.

## Actual hyperlinks extracted from the PDF

PDF page 11 labels these links "Open resource"; they are real hyperlinks rather than
missing URLs. The curriculum root was checked for reachability; child lessons are
retained as references for future review, not claimed as reviewed/copied source code.

* Curriculum: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive
* Attention: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/02-self-attention-from-scratch
* Multi-head: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/03-multi-head-attention
* Positions: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/04-positional-encoding
* Full Transformer: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/05-full-transformer
* BERT MLM: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/06-bert-masked-language-modeling
* GPT CLM: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling
* Tiny Shakespeare (also p6): https://huggingface.co/datasets/karpathy/tiny_shakespeare
* Hosting example (p2): https://render.com/
* Article platform (p3): https://medium.com/

The actual dataset card was checked in Phase 1: licensing information is incomplete,
and its dataset viewer reports legacy scripts unsupported. Before Phase 5 downloading,
verify upstream corpus rights and a pinned accessible raw data route; do not assume
the legacy load_dataset example works or reuse another assignment's corpus.
No dataset was downloaded here. See the source card:
https://huggingface.co/datasets/karpathy/tiny_shakespeare.

## Future attribution gate

For every later equation/tutorial/adapted implementation/figure record original author,
URL/version, exact adaptation and license/rights. Add primary Transformer/BERT/GPT
papers when actually reviewed. Zaid/team must understand all components and review
manual arithmetic. A source bibliography never substitutes for real measured evidence.

## Phase 2 reviewed sources and independent work

* Current Desktop handoff is preserved byte-for-byte in `phase2_instructions.txt`.
  Root `instructions.txt` remains the historical Phase 1 document. The supplied
  independent master remains Version 1.0; no Version 1.1 document was invented.
* Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017),
  **Attention Is All You Need**, arXiv:1706.03762, reviewed HTML revision v7:
  https://arxiv.org/html/1706.03762v7. §3.2.1 supplies scaled dot-product mathematics
  and its variance motivation; §4/Table 1 supports the idealized sequential-depth
  comparison. Our recurrence, explicit cost assumptions and bounded MAC arithmetic
  are reasoned educational analysis, not reproduced paper benchmark results.
* PyTorch 2.10 softmax API: https://docs.pytorch.org/docs/2.10/generated/torch.softmax.html.
  Elementary tensor matmul/masking/softmax and autograd are used; no model source
  or fused attention implementation was copied or called.
* Matplotlib 3.10.8 package metadata: https://pypi.org/project/matplotlib/3.10.8/;
  savefig API: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html.
  Compatible cp314 Windows wheel installed; other supported CI versions checked
  separately. A development dependency, not a training dependency or speed result.

Q/K/V are the authorized handoff's proposed simple example. All exact arithmetic is
written out independently in `manual_attention.md`; scalar loops reconstruct it,
then explicit PyTorch attention independently corroborates the results. No handwritten
artifact, personal attestation or trained semantic result is fabricated. F01/F02 are
original programmatic figures from the scripts, with content/source SHA256 receipts.
Zaid's personal review remains required. PDF curriculum code and any unrelated
assignment checkout were not consulted or copied in this phase.

## Phase 3 reviewed sources and independent work

* The 11-page original assignment PDF remains authoritative, particularly lessons
  03/04, M2, component/configuration tables and F03/F04. Its bytes are unchanged.
  Current Desktop instructions are preserved exactly in `phase3_instructions.txt`.
* [Vaswani et al., Attention Is All You Need, §3.2.2 and §3.5](https://arxiv.org/html/1706.03762v7)
  supplies multi-head and sinusoidal equations. No model source or paper performance
  result was copied. These educational modules have not been language-model trained.
* [PyTorch 2.10 Module.register_buffer](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
  documents persistent state/device behavior. Basic Linear, Dropout, tensor operations
  and autograd are used; high-level/fused attention classes are not called.

Module code, fixed asymmetric example matrices, Python-loop projection/head/output
oracle, full scalar sinusoidal verification, tests, runtime trace and F04 plotting
are original work in this repository. The scalar attention oracle is the existing
independently verified Phase 2 project oracle; the production engine is unchanged.
No external code template or unrelated assignment artifact was used. F03/F04 receipts
identify exact source/artifact bytes, commands and mathematical rather than trained
evidence. Zaid's personal hand-calculation review remains pending.
