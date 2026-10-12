# Three-token attention: explicit worked calculation

**Requires Zaid's mathematical review and understanding before submission.** This is
an auditable typed worked solution prepared with Codex assistance, not a handwritten
scan or evidence of Zaid's personal calculation. Reproduce these steps yourself and
explain them before attesting to understanding. No signature or attestation is implied.

The authoritative assignment is A1_AIE.pdf §6. The mathematical operation follows
[Vaswani et al., Attention Is All You Need, §3.2.1](https://arxiv.org/abs/1706.03762).
The exact expressions below are the derivation; decimals are independently calculated
with Python's scalar `math` operations, then checked against explicit PyTorch operations.
Displayed matrices use six decimal places; the verification appendix uses nine.
Compute with unrounded values, not the displayed approximations.

## Representations and operations

In a complete model, input representations X are transformed by learned projections
Q = X W_Q, K = X W_K, V = X W_V. Queries ask for compatible information, keys provide
matching features, and values carry information to aggregate. This phase supplies
Q/K/V directly; learned projections belong to Phase 3.

For each query, compute scores S = QKᵀ / sqrt(d_k), then A = softmax(S + M) across
the key axis, and O = AV. M is zero at allowed keys and negative infinity at blocked
keys. Scores are unnormalized compatibility numbers; weights are normalized numbers;
output rows are contextualized value representations. None is the Q/K/V projection
itself. Attention weights alone do not establish an explanation of semantic understanding.

For independent zero-mean, unit-variance query/key components, a dot product of d_k
terms has variance d_k. Dividing by sqrt(d_k) keeps its variance near one, reducing
extreme logits and saturated softmax as width grows. Independence/unit variance are
motivating assumptions, not guaranteed properties of every learned representation.
The divisor is feature width, not sequence length.

## 1. Exactly three tokens; explicit Q, K, V

Rows correspond to tokens t1, t2, t3. d_k = 2 and d_v = 2.

```text
Q (3×2) = [1 0]    K (3×2) = [1 0]    V (3×2) = [1 2]
          [0 1]              [0 1]              [3 0]
          [1 1]              [1 1]              [0 4]

Kᵀ (2×3) = [1 0 1]
            [0 1 1]
```

These deliberately simple numeric features are not trained token embeddings.

## 2. All nine dot products

| Query/key | Arithmetic | Result |
| --- | --- | --- |
| t1/t1 | 1×1 + 0×0 | 1 |
| t1/t2 | 1×0 + 0×1 | 0 |
| t1/t3 | 1×1 + 0×1 | 1 |
| t2/t1 | 0×1 + 1×0 | 0 |
| t2/t2 | 0×0 + 1×1 | 1 |
| t2/t3 | 0×1 + 1×1 | 1 |
| t3/t1 | 1×1 + 1×0 | 1 |
| t3/t2 | 1×0 + 1×1 | 1 |
| t3/t3 | 1×1 + 1×1 | 2 |

```text
QKᵀ (3×3) = [1 0 1]
             [0 1 1]
             [1 1 2]
```

## 3. All nine scaled scores

Let a = 1/sqrt(2) = 0.7071067811865475. Then 2a = sqrt(2).

| Query/key | Exact score | Six decimals |
| --- | --- | --- |
| t1/t1 | 1/sqrt(2) | 0.707107 |
| t1/t2 | 0/sqrt(2) | 0.000000 |
| t1/t3 | 1/sqrt(2) | 0.707107 |
| t2/t1 | 0/sqrt(2) | 0.000000 |
| t2/t2 | 1/sqrt(2) | 0.707107 |
| t2/t3 | 1/sqrt(2) | 0.707107 |
| t3/t1 | 1/sqrt(2) | 0.707107 |
| t3/t2 | 1/sqrt(2) | 0.707107 |
| t3/t3 | 2/sqrt(2) | 1.414214 |

```text
S (3×3) = [a  0  a ] ≈ [0.707107 0.000000 0.707107]
          [0  a  a ]   [0.000000 0.707107 0.707107]
          [a  a  2a]   [0.707107 0.707107 1.414214]
```

## 4. Row softmax: every numerator and denominator

Let r = exp(a) = 2.028114981647472 and exp(2a) = r² = 4.113250378782927.
For a row s, A_j = exp(s_j) / sum_l exp(s_l). Each row distributes one query's
weight among keys. Applying softmax down the query axis would instead normalize
different queries together, which is a different operation.

* Row t1 exponentials: [exp(a), exp(0), exp(a)] = [r, 1, r].
  D1 = r + 1 + r = 5.056229963294944.
  A11 = r/D1 = 0.401112092680; A12 = 1/D1 = 0.197775814640;
  A13 = r/D1 = 0.401112092680. Sum = (2r+1)/D1 = 1 exactly
  (displayed 12-digit values sum to 1.000000000000).
* Row t2 exponentials: [exp(0), exp(a), exp(a)] = [1, r, r].
  D2 = 1 + r + r = 5.056229963294944.
  A21 = 1/D2 = 0.197775814640; A22 = r/D2 = 0.401112092680;
  A23 = r/D2 = 0.401112092680. Sum = (1+2r)/D2 = 1 exactly
  (displayed sum 1.000000000000).
* Row t3 exponentials: [exp(a), exp(a), exp(2a)] = [r, r, r²].
  D3 = r + r + r² = 8.169480342077872.
  A31 = r/D3 = 1/(2+r) = 0.248255078258;
  A32 = r/D3 = 1/(2+r) = 0.248255078258;
  A33 = r²/D3 = r/(2+r) = 0.503489843485.
  Sum = (2r+r²)/D3 = 1 exactly (displayed sum 1.000000000001).

Define p = r/(2r+1), s = 1/(2r+1), u = 1/(2+r), w = r/(2+r).

```text
A (3×3) = [p s p] ≈ [0.401112 0.197776 0.401112]
          [s p p]   [0.197776 0.401112 0.401112]
          [u u w]   [0.248255 0.248255 0.503490]
```

## 5. Weighted values: every output component

* O1 = p[1,2] + s[3,0] + p[0,4].
  O11 = p×1 + s×3 + p×0 = p+3s = 0.994439536601.
  O12 = p×2 + s×0 + p×4 = 6p = 2.406672556079.
* O2 = s[1,2] + p[3,0] + p[0,4].
  O21 = s×1 + p×3 + p×0 = s+3p = 1.401112092680.
  O22 = s×2 + p×0 + p×4 = 2s+4p = 2.000000000000.
* O3 = u[1,2] + u[3,0] + w[0,4].
  O31 = u×1 + u×3 + w×0 = 4u = 0.993020313031.
  O32 = u×2 + u×0 + w×4 = 2u+4w = 2.510469530454.

```text
O = AV (3×2) ≈ [0.994440 2.406673]
               [1.401112 2.000000]
               [0.993020 2.510470]
```

## 6. Causal masking before softmax

The query/key indices are aligned. The diagonal is allowed. Boolean True in the
implementation means forbidden; its additive equivalent is negative infinity.

```text
Original S (3×3) = [a 0 a ]    M (3×3) = [0 -∞ -∞]
                  [0 a a ]              [0  0 -∞]
                  [a a 2a]              [0  0  0]

S + M (3×3) = [a -∞ -∞] ≈ [0.707107       -∞       -∞]
              [0  a -∞]   [0.000000 0.707107       -∞]
              [a  a 2a]   [0.707107 0.707107 1.414214]
```

exp(-∞) = 0, so forbidden keys have exactly zero weight.

* Row t1 exponentials [r, 0, 0]; denominator r = 2.028114981647472.
  Probabilities [r/r, 0/r, 0/r] = [1,0,0]; row sum = 1.
* Row t2 exponentials [1, r, 0]; denominator 1+r = 3.028114981647472.
  Probabilities [1/(1+r), r/(1+r), 0/(1+r)] =
  [0.330238450673, 0.669761549327, 0]; row sum = 1.000000000000.
* Row t3 exponentials [r, r, r²]; denominator 8.169480342077872.
  Probabilities [r/D3, r/D3, r²/D3] =
  [0.248255078258, 0.248255078258, 0.503489843485];
  exact sum = 1; rounded sum = 1.000000000001.

Let c = 1/(1+r), b = r/(1+r).

```text
A_c (3×3) = [1 0 0] ≈ [1.000000 0.000000 0.000000]
            [c b 0]   [0.330238 0.669762 0.000000]
            [u u w]   [0.248255 0.248255 0.503490]
```

All six causal output components:

* O_c1 = 1[1,2] + 0[3,0] + 0[0,4].
  O_c11 = 1×1+0×3+0×0 = 1; O_c12 = 1×2+0×0+0×4 = 2.
* O_c2 = c[1,2] + b[3,0] + 0[0,4].
  O_c21 = c×1+b×3+0×0 = (1+3r)/(1+r) = 2.339523098653.
  O_c22 = c×2+b×0+0×4 = 2/(1+r) = 0.660476901347.
* O_c3 = u[1,2] + u[3,0] + w[0,4].
  O_c31 = u×1+u×3+w×0 = 4u = 0.993020313031.
  O_c32 = u×2+u×0+w×4 = 2u+4w = 2.510469530454.

```text
O_c = A_c V (3×2) ≈ [1.000000 2.000000]
                    [2.339523 0.660477]
                    [0.993020 2.510470]
```

## 7. What each position can know

Unmasked attention uses every value for every query: bidirectional encoder-style
visibility. Causal t1 uses only V1; t2 mixes V1 and V2; t3 can mix V1,V2,V3.
The final row is unchanged because no future key exists. Modifying future K/V
cannot change earlier causal outputs; tests check this and future-value gradients.

An autoregressive model predicts the next token using the prefix. Future information
would leak the answer during teacher-forced training and is unavailable during
generation. Causal masking lets all *known training positions* be computed in parallel
without that leak. Generation still produces tokens sequentially, because the next
input token has not yet been sampled. These numeric weights do not show trained
language knowledge; no model has been trained.

## Independent numerical verification appendix

The JSON block is a machine-checkable transcription of the derived values, rounded
to nine decimals. `python -m experiments.attention_example --figure` compares ALL
seven intermediates to separate scalar loops with absolute tolerance 5.1e-10
(nine-decimal rounding plus floating error). It then compares the production float64
engine at rtol=1e-12, atol=1e-12. JSON null denotes a forbidden negative-infinity
score, not missing data. The reference uses Python `math`, not torch softmax/matmul
or the production mask helper. Tests also verify unrounded random examples.

<!-- attention-reference -->
```json
{
  "dot_products": [[1,0,1],[0,1,1],[1,1,2]],
  "scores": [[0.707106781,0,0.707106781],[0,0.707106781,0.707106781],[0.707106781,0.707106781,1.414213562]],
  "weights": [[0.401112093,0.197775815,0.401112093],[0.197775815,0.401112093,0.401112093],[0.248255078,0.248255078,0.503489843]],
  "output": [[0.994439537,2.406672556],[1.401112093,2],[0.993020313,2.510469530]],
  "masked_scores": [[0.707106781,null,null],[0,0.707106781,null],[0.707106781,0.707106781,1.414213562]],
  "masked_weights": [[1,0,0],[0.330238451,0.669761549,0],[0.248255078,0.248255078,0.503489843]],
  "masked_output": [[1,2],[2.339523099,0.660476901],[0.993020313,2.510469530]]
}
```

F01 at `report/evidence/F01_manual_attention.png` is a programmatically rendered
summary of the verified derivation; this document contains the full arithmetic.
F02 at `report/evidence/F02_causal_mask.png` shows structural visibility, not trained
attention (F07). JSON verification receipts and SHA256 provenance accompany them.
The programs supplement the derivation and do not satisfy Zaid's personal review.
