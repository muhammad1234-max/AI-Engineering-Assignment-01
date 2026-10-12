"""Educational comparison of MLM (BERT-style) vs CLM (GPT-style) objectives.

This experiment demonstrates the fundamental difference between bidirectional
masked language modeling and causal language modeling using small attention
matrices and mask visualizations. No full BERT training is performed — this
is a conceptual/mathematical comparison as required by the assignment.
"""

import json
import sys

import torch

from zaid_lm.attention import scaled_dot_product_attention


def clm_mask_matrix(seq_len):
    """Causal LM (GPT-style): each token sees only itself and past tokens.

    Token i can attend to tokens [0, 1, ..., i].
    Visibility matrix: lower-triangular ones.
    """
    # True = visible (1), False = blocked (0)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def mlm_mask_matrix(seq_len):
    """Masked LM (BERT-style): each token sees ALL tokens (bidirectional).

    Every token can attend to every other token.
    Visibility matrix: all ones.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool)


def demonstrate_visibility():
    """Show what each token can see under CLM vs MLM for a 5-token sequence."""
    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)

    clm = clm_mask_matrix(seq_len)
    mlm = mlm_mask_matrix(seq_len)

    results = {
        "tokens": tokens,
        "sequence_length": seq_len,
        "CLM_visibility": {},
        "MLM_visibility": {},
    }

    print("=" * 60)
    print("CLM (GPT-style) — Causal Language Modeling")
    print("=" * 60)
    print("Each token can ONLY see itself and past tokens.\n")
    print("Visibility matrix (1 = can see, 0 = blocked):")
    print(clm.int().tolist())
    print()
    for i, token in enumerate(tokens):
        visible = [tokens[j] for j in range(seq_len) if clm[i, j]]
        results["CLM_visibility"][token] = visible
        print(f"  Token '{token}' (pos {i}) sees: {visible}")

    print("\n  Objective: predict next token x_(t+1) given x_(0..t)")
    print("  Loss: CE(logits[t], target[t+1]) for each position")

    print()
    print("=" * 60)
    print("MLM (BERT-style) — Masked Language Modeling")
    print("=" * 60)
    print("Each token can see ALL tokens (bidirectional).\n")
    print("Visibility matrix (1 = can see, 0 = blocked):")
    print(mlm.int().tolist())
    print()
    for i, token in enumerate(tokens):
        visible = [tokens[j] for j in range(seq_len) if mlm[i, j]]
        results["MLM_visibility"][token] = visible
        print(f"  Token '{token}' (pos {i}) sees: {visible}")

    print("\n  Objective: randomly mask ~15% of tokens, predict original token")
    print("  Loss: CE(logits[masked_pos], original_token) only at masked positions")

    return results


def attention_comparison():
    """Run actual attention with both mask types and compare outputs.

    Uses the same Q, K, V but different visibility constraints to show
    how causal vs bidirectional masking changes attention weights and outputs.
    """
    torch.manual_seed(42)
    seq_len = 5
    d_k = 4

    # Same Q, K, V for both
    q = torch.randn(1, seq_len, d_k, dtype=torch.float64)
    k = torch.randn(1, seq_len, d_k, dtype=torch.float64)
    v = torch.randn(1, seq_len, d_k, dtype=torch.float64)

    # CLM: causal attention (upper triangle blocked)
    clm_output, clm_weights = scaled_dot_product_attention(q, k, v, causal=True)

    # MLM: full bidirectional attention (no mask)
    mlm_output, mlm_weights = scaled_dot_product_attention(q, k, v, causal=False)

    results = {
        "d_k": d_k,
        "seq_len": seq_len,
        "CLM_attention_weights": clm_weights[0].tolist(),
        "MLM_attention_weights": mlm_weights[0].tolist(),
        "weight_difference_max": (clm_weights - mlm_weights).abs().max().item(),
        "output_difference_max": (clm_output - mlm_output).abs().max().item(),
    }

    print("\n" + "=" * 60)
    print("Attention Weight Comparison (same Q, K, V)")
    print("=" * 60)

    print("\nCLM (causal) attention weights:")
    for i in range(seq_len):
        row = [f"{clm_weights[0, i, j]:.4f}" for j in range(seq_len)]
        print(f"  token {i}: [{', '.join(row)}]")

    print("\nMLM (bidirectional) attention weights:")
    for i in range(seq_len):
        row = [f"{mlm_weights[0, i, j]:.4f}" for j in range(seq_len)]
        print(f"  token {i}: [{', '.join(row)}]")

    print(f"\nMax weight difference: {results['weight_difference_max']:.6f}")
    print(f"Max output difference: {results['output_difference_max']:.6f}")

    # Key observations
    print("\n" + "=" * 60)
    print("Key Differences")
    print("=" * 60)
    print("""
1. CAUSAL (CLM/GPT):
   - Token 0 only attends to itself (row 0 has weight only at col 0)
   - Each subsequent token distributes attention over more keys
   - Future tokens are ALWAYS zero weight (strictly upper triangle)
   - Used for GENERATION: predict next token autoregressively

2. BIDIRECTIONAL (MLM/BERT):
   - Every token attends to ALL tokens including future ones
   - Attention weights are more distributed across all positions
   - Token 0 can see information from token 4 (impossible in CLM)
   - Used for UNDERSTANDING: fill in masked tokens using full context

3. TRAINING OBJECTIVE:
   - CLM: shifted targets, x[t] -> predict x[t+1], every position is a loss term
   - MLM: random ~15% masking, predict original token ONLY at masked positions
   - CLM enables autoregressive generation; MLM does not directly

4. INFORMATION FLOW:
   - In CLM, information flows left-to-right only (causal)
   - In MLM, information flows in all directions (bidirectional)
   - This means BERT understands context better, but GPT can generate text
""")

    return results


def shifted_target_demo():
    """Demonstrate the CLM shifted-target training scheme."""
    print("=" * 60)
    print("CLM Shifted Target Training Scheme")
    print("=" * 60)

    tokens = ["<s>", "The", "cat", "sat", "on"]
    ids = list(range(len(tokens)))

    print(f"\nFull sequence: {tokens}")
    print(f"Token IDs:     {ids}")
    print()
    print("Training pairs (input -> target):")
    print(f"  Input (x):  {tokens[:-1]}  = ids {ids[:-1]}")
    print(f"  Target (y): {tokens[1:]}   = ids {ids[1:]}")
    print()
    print("At each position t:")
    for t in range(len(tokens) - 1):
        print(
            f"  t={t}: input='{tokens[t]}' (id={ids[t]}), "
            f"predict='{tokens[t + 1]}' (id={ids[t + 1]})"
        )
    print()
    print("Loss = mean( CE(logits[t], target_id[t]) ) for t = 0..T-2")
    print("The model sees x[0..t] and must predict x[t+1] at each step.")


def main():
    print("MLM vs CLM — Educational Comparison Experiment")
    print("(No full BERT training performed; conceptual demonstration only)")
    print()

    visibility = demonstrate_visibility()
    print()
    shifted_target_demo()
    print()
    attn_results = attention_comparison()

    # Merge results for evidence
    combined = {
        "status": "PASS",
        "description": "Educational MLM vs CLM comparison — no BERT training performed",
        **visibility,
        "attention_comparison": attn_results,
    }

    if "--json" in sys.argv:
        print(json.dumps(combined, indent=2, default=str))

    return combined


if __name__ == "__main__":
    main()
