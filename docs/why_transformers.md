# Why Transformers: a reasoned comparison

This is a complexity analysis, **not measured training speed or memory**. The
assignment permits a reasoned comparison (A1_AIE.pdf §5.1). Primary reference:
[Vaswani et al., §3.2 and §4/Table 1](https://arxiv.org/abs/1706.03762).
There is no trained RNN or Transformer, timing benchmark, or GPU result in this phase.

## Dependence and sequential depth

A simple recurrent layer computes h_t = phi(W_x x_t + W_h h_(t-1) + b).
Even with all training inputs known, h_t needs the already computed h_(t-1).
An ordinary recurrence therefore has T serial state updates per layer for T tokens.
Matrix operations within an update and independent batch examples can be parallelized;
special parallelizable recurrent formulations are outside this comparison.

An attention layer projects all supplied positions to Q/K/V, forms QKᵀ/sqrt(d_k),
masks future keys, applies row softmax, then computes AV. Dense tensor operations
can handle positions together. Teacher forcing supplies the known training sequence;
the causal mask ensures position t only uses positions <= t. A future decoder will
compare its logits at t to target token t+1. The mask alone does not implement loss.

For a fixed number L of layers, there are L serial layer stages; Transformer training
does not add a recurrent chain of T hidden-state transitions within each stage.
The often quoted O(1) sequential operations per attention layer describes dependencies
*across positions*, not constant wall time: matrix reductions, synchronization and
hardware scheduling still take work and depth. A legal distant key can directly
contribute to a query in one attention layer; an RNN transfers information through
the intervening state transitions. A causal query cannot connect to future keys.

Autoregressive generation still samples one new token before computing the next.
Caching can avoid recomputing prior keys/values but cannot remove that dependence.
No caching or generation is implemented here.

## Costs under explicit assumptions

Assume dense full attention, sequence T, batch B, width d, FFN width f, L layers;
query/key/value projections have width d and total head width d. RNN input/state
widths are both d. Count multiply-accumulate operations (MACs), ignoring biases,
softmax/nonlinearity/dropout/normalization, and treating a MAC as one operation.
No sparse/fused memory-saving kernel or hardware-specific speed is assumed.

| Per layer | Ordinary simple RNN | Dense causal Transformer |
| --- | --- | --- |
| Training serial position dependence | T recurrent updates | Positions handled together; serial layer stages remain |
| Dense matrix MACs | About 2BTd² for input and recurrent transforms | About 4BTd² for Q/K/V + output projection, 2BTdf for FFN, 2BT²d for scores + aggregation |
| Attention score/weight storage | No T×T score matrix | O(BHT²) entries for H heads in an explicit implementation |
| Other training activations | Roughly O(BTd) per layer, plus gradients and saved states | At least O(BTd + BTf) per layer, plus attention and gradients |
| Path from earlier to distant later position | Through successive hidden-state transitions | Direct permitted attention edge in one layer |
| Next-token generation | Sequential | Sequential; possible future KV-cache savings |

Constants, saved intermediates, precision, optimizer state, checkpointing and kernels
affect real peak memory. Quadratic attention storage is not the whole model memory.
The paper's idealized attention-only O(T²d) and recurrent O(Td²) comparison omits the
projection/FFN terms shown here. Causal masking in this implementation still constructs
the full square matrix; it does not halve the actual dense matmul work.

## Bounded educational arithmetic

Use proposed T=128, d=128, f=512, B=1, one layer; these are not trained measurements.

* RNN: 2×128×128² = 4,194,304 MACs.
* Q/K/V and output projections: 4×128×128² = 8,388,608 MACs.
* FFN: 2×128×128×512 = 16,777,216 MACs.
* Scores and value aggregation: 2×128²×128 = 4,194,304 MACs.
* Transformer dense total in this approximation: 29,360,128 MACs.
* Four heads store 4×128² = 65,536 score entries: 262,144 bytes for one float32
  score tensor, before separate weights, gradients, batch scaling or other activations.

Doubling T to 256 doubles projection/FFN work and quadruples score work/storage.
These counts describe different architectures, not an equal-quality experiment or
proof of comparative runtime. GPUs may execute large matrix operations efficiently
across batch/sequence positions, while a recurrence constrains available parallelism.
For short sequences, small batches, CPU execution, launch overhead or optimized
recurrent kernels, the practical comparison can change. Transformers are not always
faster and do not always use less memory. Fair measured comparisons would require
specified hardware, precision, kernels, batch, sequence, architectures and objective.

## Phase boundary

The implemented Phase 2 engine computes supplied Q/K/V attention only. The projected
Transformer costs describe the forthcoming architecture, not implemented modules.
Phase 3 will add learned multi-head projections/split/merge and sinusoidal positions;
blocks, loss, data, training and real timing measurements belong to later phases.
