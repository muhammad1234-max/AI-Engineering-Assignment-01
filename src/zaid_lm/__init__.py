"""Zaid's independently implemented educational language-model project."""

from zaid_lm.attention import causal_forbidden_mask, scaled_dot_product_attention
from zaid_lm.checkpoint import load_checkpoint, save_checkpoint
from zaid_lm.config import ProjectConfig, load_config
from zaid_lm.dataset import (
    ShakespeareDataset,
    download_tiny_shakespeare,
    get_batch,
    prepare_dataset,
)
from zaid_lm.gpt import GPT
from zaid_lm.multihead_attention import MultiHeadSelfAttention
from zaid_lm.positional_encoding import SinusoidalPositionalEncoding
from zaid_lm.tokenizer import CharacterTokenizer
from zaid_lm.train import train
from zaid_lm.transformer_block import FeedForwardNetwork, TransformerDecoderBlock

__all__ = [
    "GPT",
    "CharacterTokenizer",
    "FeedForwardNetwork",
    "MultiHeadSelfAttention",
    "ProjectConfig",
    "ShakespeareDataset",
    "SinusoidalPositionalEncoding",
    "TransformerDecoderBlock",
    "causal_forbidden_mask",
    "download_tiny_shakespeare",
    "get_batch",
    "load_checkpoint",
    "load_config",
    "prepare_dataset",
    "save_checkpoint",
    "scaled_dot_product_attention",
    "train",
]
__version__ = "0.1.0"
