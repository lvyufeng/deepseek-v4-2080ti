from __future__ import annotations

from pathlib import Path

import pytest

from src.loader.gguf.bundle import read_gguf_bundle
from src.models.minimax_m2.spec import MiniMaxM2Spec
from tests.gguf_test_utils import GGML_F32, write_gguf, write_minimax_bundle


REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def test_minimax_spec_parses_tiny_bundle_and_validates_schema(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=2)
    bundle = read_gguf_bundle(root)
    spec = MiniMaxM2Spec()

    params = spec.parse_params(bundle)
    validation = spec.validate_bundle(bundle)

    assert params.architecture == "minimax-m2"
    assert params.n_layers == 2
    assert params.hidden_size == 8
    assert params.vocab_size == 16
    assert params.n_heads == 2
    assert params.n_kv_heads == 1
    assert params.head_dim == 4
    assert params.n_routed_experts == 4
    assert params.top_k == 2
    assert params.expert_intermediate_size == 4
    assert params.attention_kind == "gqa_separate_qkv"
    assert params.gate_function == "gguf_enum:2"
    assert validation.ok, validation.errors
    assert validation.mapped_sources == 29
    assert validation.role_counts["routed_w1"] == 2
    assert validation.role_counts["routed_w2"] == 2
    assert validation.role_counts["routed_w3"] == 2


def test_minimax_capability_marks_runtime_deferred_but_placement_candidate(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)
    bundle = read_gguf_bundle(root)
    report = MiniMaxM2Spec().capability_report(bundle, gpu_count=4, gpu_memory_gib=22.0)

    caps = {item.name: item for item in report.capabilities}
    placements = {item.name: item for item in report.placements}

    assert caps["embedding:q4_k"].status == "supported"
    assert caps["attn_q:q5_k"].status == "supported"
    assert caps["routed_w1:iq2_xxs"].status == "supported"
    assert caps["generation"].status == "candidate"
    assert "MiniMax-M2" in caps["generation"].reason
    assert report.tensor_type_counts["iq2_xxs"] == 3
    assert report.tensor_type_counts["q5_k"] == 4
    assert placements["all_device_lowbit"].status == "candidate"
    assert placements["heterogeneous_routed_experts"].status == "candidate"


def test_minimax_spec_does_not_require_deepseek_v4_only_tensors(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)
    bundle = read_gguf_bundle(root)
    validation = MiniMaxM2Spec().validate_bundle(bundle)

    assert validation.ok, validation.errors
    joined = "\n".join(validation.errors)
    assert "attn_q_a" not in joined
    assert "attn_kv" not in joined
    assert "ffn_gate_tid2eid" not in joined
    assert "ffn_gate_shexp" not in joined
    assert "hc_" not in joined
    assert "indexer" not in joined


def test_minimax_shape_validation_catches_wrong_tensor_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    metadata = {
        "general.architecture": "minimax-m2",
        "minimax-m2.block_count": 1,
        "minimax-m2.embedding_length": 8,
        "minimax-m2.vocab_size": 16,
        "minimax-m2.context_length": 128,
        "minimax-m2.attention.head_count": 2,
        "minimax-m2.attention.head_count_kv": 1,
        "minimax-m2.attention.key_length": 4,
        "minimax-m2.expert_count": 4,
        "minimax-m2.expert_used_count": 2,
        "minimax-m2.expert_feed_forward_length": 4,
    }
    write_gguf(path, metadata=metadata, tensors=[("token_embd.weight", (7, 16), GGML_F32)])
    bundle = read_gguf_bundle(path)

    validation = MiniMaxM2Spec().validate_bundle(bundle)

    assert not validation.ok
    assert any("token_embd.weight unexpected shape" in error for error in validation.errors)


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_m27_bundle_validates_when_available() -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    spec = MiniMaxM2Spec()
    params = spec.parse_params(bundle)
    validation = spec.validate_bundle(bundle)
    report = spec.capability_report(bundle)

    assert params.n_layers == 62
    assert params.hidden_size == 3072
    assert params.vocab_size == 200064
    assert params.context_length == 196608
    assert params.n_routed_experts == 256
    assert params.top_k == 8
    assert bundle.tensor_count == 809
    assert validation.ok, validation.errors[:20]
    assert report.tensor_type_counts["iq2_xxs"] == 62 * 3
    assert any(item.name == "generation" and item.status == "candidate" for item in report.capabilities)
