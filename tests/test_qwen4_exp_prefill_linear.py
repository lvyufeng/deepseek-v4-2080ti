"""Tests for the optional SM75 FP16 prefill projection path."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.qwen4_exp.layers import prefill_linear


def test_prefill_linear_cpu_is_exact(monkeypatch) -> None:
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "1")
    x = torch.randn(9, 32, dtype=torch.bfloat16)
    weight = torch.randn(17, 32, dtype=torch.bfloat16)

    got = prefill_linear(x, weight)
    want = F.linear(x, weight)

    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_prefill_linear_small_cuda_rows_keep_bf16(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")
    x = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16)

    got = prefill_linear(x, weight)
    want = F.linear(x, weight)

    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_prefill_linear_large_cuda_rows_use_close_fp16(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "8")
    torch.manual_seed(20260830)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(48, 64, device="cuda", dtype=torch.bfloat16) * 0.05

    got = prefill_linear(x, weight)
    want = F.linear(x, weight)

    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


def test_prefill_linear_fp16_fp32_output_matches_explicit_mm(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_FP32_OUTPUT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "8")
    torch.manual_seed(20260831)
    x = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(48, 64, device="cuda", dtype=torch.bfloat16) * 0.05

    got = prefill_linear(x, weight)
    explicit = torch.mm(
        x.reshape(-1, 64).to(torch.float16),
        weight.to(torch.float16).t(),
        out_dtype=torch.float32,
    ).to(torch.bfloat16).reshape(2, 4, 48)

    torch.testing.assert_close(got, explicit, rtol=0, atol=0)


def test_prefill_linear_fp16_fp32_output_keeps_small_rows_bf16(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_FP32_OUTPUT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")
    x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(48, 64, device="cuda", dtype=torch.bfloat16)

    got = prefill_linear(x, weight)
    want = F.linear(x, weight)

    torch.testing.assert_close(got, want, rtol=0, atol=0)
