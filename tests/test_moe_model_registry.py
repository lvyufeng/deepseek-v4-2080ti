from __future__ import annotations

from pathlib import Path

import pytest

from src.loader.gguf.bundle import read_gguf_bundle
from src.components.moe.registry import detect_spec, get_spec, known_architectures
from tests.gguf_test_utils import write_gguf, write_minimax_bundle


def test_known_architectures_include_deepseek_v4_and_minimax() -> None:
    assert "deepseek4" in known_architectures()
    assert "minimax-m2" in known_architectures()


def test_get_spec_normalizes_architecture() -> None:
    assert get_spec("MiniMax-M2").architecture == "minimax-m2"
    assert get_spec("deepseek4").architecture == "deepseek4"


def test_detect_spec_from_minimax_bundle(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)
    bundle = read_gguf_bundle(root)

    spec = detect_spec(bundle)

    assert spec.architecture == "minimax-m2"


def test_detect_spec_from_deepseek_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ds4.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "deepseek4", "deepseek4.block_count": 1},
        tensors=[],
    )
    bundle = read_gguf_bundle(path)

    spec = detect_spec(bundle)

    assert spec.architecture == "deepseek4"


def test_unknown_architecture_error(tmp_path: Path) -> None:
    path = tmp_path / "unknown.gguf"
    write_gguf(path, metadata={"general.architecture": "dense-thing"}, tensors=[])
    bundle = read_gguf_bundle(path)

    with pytest.raises(ValueError, match="unsupported MoE architecture"):
        detect_spec(bundle)
