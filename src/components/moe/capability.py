from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantCapability:
    type_name: str
    header_supported: bool
    payload_reader_supported: bool
    routed_raw_supported: bool
    runtime_supported: bool
    notes: str


_CAPABILITIES = {
    "f32": QuantCapability("f32", True, True, False, True, "dense scalar tensor payload supported"),
    "f16": QuantCapability("f16", True, True, False, True, "dense scalar tensor payload supported"),
    "bf16": QuantCapability("bf16", True, True, False, True, "dense scalar tensor payload supported"),
    "i32": QuantCapability("i32", True, True, False, True, "dense scalar tensor payload supported"),
    "q8_0": QuantCapability("q8_0", True, True, True, True, "raw q8_0 binding exists for supported modules"),
    "q2_k": QuantCapability("q2_k", True, True, True, True, "routed expert raw blocks supported in existing DeepSeek-V4 paths"),
    "q3_k": QuantCapability("q3_k", True, True, False, False, "GGUF q3_k payload can be inventoried; CUDA runtime support is deferred"),
    "iq2_xxs": QuantCapability("iq2_xxs", True, True, True, True, "routed expert raw blocks supported by MiniMax/DeepSeek-V4 GGUF CUDA paths"),
    "iq2_xs": QuantCapability("iq2_xs", True, True, True, True, "raw iq2_xs CUDA block-dot GEMM supported (type_id 5)"),
    "iq3_xxs": QuantCapability("iq3_xxs", True, True, True, True, "raw iq3_xxs CUDA block-dot GEMM supported (type_id 6)"),
    "iq1_m": QuantCapability("iq1_m", True, True, True, True, "routed expert raw blocks supported in existing DeepSeek-V4 paths"),
    "iq4_xs": QuantCapability("iq4_xs", True, True, True, True, "raw iq4_xs CUDA block-dot GEMM supported (type_id 7)"),
    "q4_k": QuantCapability("q4_k", True, True, True, True, "raw q4_k CUDA GEMM and selected-row embedding supported"),
    "q5_k": QuantCapability("q5_k", True, True, True, True, "raw q5_k CUDA GEMM supported"),
    "q6_k": QuantCapability("q6_k", True, True, True, True, "raw q6_k CUDA block-dot GEMM supported (type_id 8)"),
}


def quant_capability(type_name: str) -> QuantCapability:
    return _CAPABILITIES.get(
        type_name,
        QuantCapability(type_name, False, False, False, False, "unknown GGUF tensor type"),
    )


def capability_status_for_role(role: str, type_name: str, *, architecture: str) -> tuple[str, str]:
    cap = quant_capability(type_name)
    if architecture == "glm-dsa":
        if cap.runtime_supported:
            return "supported", f"{cap.notes}; runs through the GLM-DSA GGUF runtime"
        if cap.header_supported and cap.payload_reader_supported:
            return "deferred", f"{cap.notes}; GLM-DSA runtime path is deferred"
        return "unsupported", cap.notes
    if architecture == "minimax-m2":
        if role in {"routed_w1", "routed_w2", "routed_w3"} and type_name == "iq2_xxs":
            return "supported", "iq2_xxs routed blocks are readable and run through MiniMax TP CUDA grouped MoE"
        if role in {"attn_q", "attn_k", "attn_v", "attn_o"} and type_name == "q5_k":
            return "supported", "q5_k attention projections run through raw-block CUDA GEMM"
        if role in {"embedding", "lm_head"} and type_name == "q4_k":
            return "supported", "q4_k embedding/head run through raw-block CUDA kernels"
        if type_name == "f32":
            return "supported", "f32 tensor payload is supported by the MiniMax runtime"
    if cap.runtime_supported:
        return "supported", cap.notes
    if cap.header_supported:
        return "deferred", cap.notes
    return "unsupported", cap.notes
