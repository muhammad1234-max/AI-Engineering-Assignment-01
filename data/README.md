# Tiny Shakespeare data plan (not downloaded)

PDF p.6 section 9 links to https://huggingface.co/datasets/karpathy/tiny_shakespeare.
In Phase 5 inspect and record dataset-card license, upstream attribution, pinned revision,
download URL, format conversion, exact bytes/lines and SHA256. Do not reuse another
assignment's cached corpus, tokens or hashes. PDF approximate size/lines is background,
not a local measurement.

Freeze deterministic disjoint train/validation/test splits before fitting. Review
duplicates and boundaries. Fit character vocabulary on training only; document
special/unknown tokens and reversible encoding. Enforce tokenizer/model vocabulary
compatibility. Shift x/y by one and prevent windows crossing splits. Validation for
selection; test only after explicitly approved frozen selection. Record filtering.
Confirm rights before download/redistribution. Actual data is ignored by Git.
