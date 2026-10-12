"""Unit tests for dataset loading, shifted targets, and batching."""

from __future__ import annotations

import pytest
import torch

from zaid_lm.dataset import ShakespeareDataset, get_batch


class TestDatasetAndBatching:
    def test_shifted_targets_alignment(self):
        """Verify that y[t] is exactly x[t+1] (next-token prediction target)."""
        # Tokens: [0, 1, 2, 3, 4, 5, 6, 7]
        tokens = torch.arange(8, dtype=torch.long)
        context_length = 4
        dataset = ShakespeareDataset(tokens, context_length=context_length)

        # Expected samples: len(tokens) - context_length = 8 - 4 = 4
        assert len(dataset) == 4

        # Sample 0: x = [0, 1, 2, 3], y = [1, 2, 3, 4]
        x0, y0 = dataset[0]
        assert torch.equal(x0, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(y0, torch.tensor([1, 2, 3, 4]))
        assert torch.equal(x0[1:], y0[:-1])  # Shift property

        # Sample 3: x = [3, 4, 5, 6], y = [4, 5, 6, 7]
        x3, y3 = dataset[3]
        assert torch.equal(x3, torch.tensor([3, 4, 5, 6]))
        assert torch.equal(y3, torch.tensor([4, 5, 6, 7]))

    def test_invalid_tokens_rejected(self):
        with pytest.raises(ValueError, match="1D tensor"):
            ShakespeareDataset(torch.zeros(2, 4), context_length=2)

        with pytest.raises(ValueError, match="positive"):
            ShakespeareDataset(torch.arange(10), context_length=0)

        with pytest.raises(ValueError, match="greater than context_length"):
            ShakespeareDataset(torch.arange(4), context_length=4)

    def test_get_batch_shapes_and_types(self):
        data = torch.arange(100, dtype=torch.long)
        batch_size = 4
        context_length = 16

        x, y = get_batch(data, batch_size, context_length)
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

        # For every item in the batch, y must equal x shifted by 1
        for i in range(batch_size):
            assert torch.equal(x[i, 1:], y[i, :-1])

    def test_get_batch_deterministic_with_generator(self):
        data = torch.arange(50, dtype=torch.long)
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)

        x1, y1 = get_batch(data, 4, 8, generator=gen1)
        x2, y2 = get_batch(data, 4, 8, generator=gen2)

        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
