"""Correctness tests for the memory-bounded QSA Python fallback."""

from __future__ import annotations

import copy

import torch

from src.models.qwen4_exp.attention import QSAAttention
from src.models.qwen4_exp.config import Qwen4ExpTextConfig


def _weights(config: Qwen4ExpTextConfig) -> dict[str, torch.Tensor]:
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim * 2
    kv = config.num_key_value_heads * config.head_dim
    index = (
        (config.indexer_n_heads or 0) * (config.indexer_head_dim or 0)
        + (config.indexer_kv_heads or 0) * (config.indexer_head_dim or 0)
    )
    return {
        "q_proj": torch.randn(query, hidden, dtype=torch.bfloat16) * 0.02,
        "k_proj": torch.randn(kv, hidden, dtype=torch.bfloat16) * 0.02,
        "v_proj": torch.randn(kv, hidden, dtype=torch.bfloat16) * 0.02,
        "o_proj": torch.randn(hidden, config.num_attention_heads * config.head_dim, dtype=torch.bfloat16) * 0.02,
        "q_norm": torch.randn(config.head_dim, dtype=torch.bfloat16) * 0.02,
        "k_norm": torch.randn(config.head_dim, dtype=torch.bfloat16) * 0.02,
        "index_qk_proj": torch.randn(index, hidden, dtype=torch.bfloat16) * 0.02,
        "q_layernorm": torch.randn(config.indexer_head_dim, dtype=torch.bfloat16) * 0.02,
        "k_layernorm": torch.randn(config.indexer_head_dim, dtype=torch.bfloat16) * 0.02,
    }


def test_qsa_chunked_fallback_matches_dense_reference(monkeypatch) -> None:
    config = Qwen4ExpTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=32,
        indexer_compress_ratio=4,
    )
    weights = _weights(config)
    hidden = torch.randn(1, 29, config.hidden_size, dtype=torch.bfloat16)
    cos = torch.ones(1, 29, config.rotary_dim, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)

    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_CUDA", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FALLBACK_ROWS", "5")
    chunked = QSAAttention(config, copy.deepcopy(weights))(
        hidden,
        cos,
        sin,
        None,
        past_len=0,
    )
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FALLBACK_ROWS", "29")
    dense = QSAAttention(config, copy.deepcopy(weights))(
        hidden,
        cos,
        sin,
        None,
        past_len=0,
    )

    torch.testing.assert_close(chunked, dense, rtol=0, atol=0)
