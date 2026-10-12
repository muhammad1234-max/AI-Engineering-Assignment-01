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
