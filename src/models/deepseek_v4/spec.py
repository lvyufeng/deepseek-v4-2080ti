from __future__ import annotations

from collections import Counter, defaultdict

from src.loader.gguf.bundle import GGUFBundle
from src.components.moe.capability import capability_status_for_role
from src.components.moe.placement import HardwareProfile, heterogeneous_expert_decision, lowbit_device_resident_decision
from src.components.moe.spec import (
    CapabilityItem,
    CapabilityReport,
    MoEArchitectureParams,
    PlacementDecision,
    SpecValidation,
    TensorMapping,
    bytes_by,
    counts_by,
    metadata_float,
    metadata_int,
)


class DeepSeekV4Spec:
    architecture = "deepseek4"
    display_name = "DeepSeek-V4"

    def parse_params(self, bundle: GGUFBundle) -> MoEArchitectureParams:
        md = bundle.metadata
        vocab = metadata_int(md, "tokenizer.ggml.tokens", metadata_int(md, "deepseek4.vocab_size", 0))
        head_dim = metadata_int(md, "deepseek4.attention.key_length", 0)
        return MoEArchitectureParams(
            architecture=self.architecture,
            n_layers=metadata_int(md, "deepseek4.block_count", 0),
            hidden_size=metadata_int(md, "deepseek4.embedding_length", 0),
            vocab_size=vocab,
            context_length=metadata_int(md, "deepseek4.context_length", 0),
            n_heads=metadata_int(md, "deepseek4.attention.head_count", 0),
            n_kv_heads=metadata_int(md, "deepseek4.attention.head_count_kv", 0),
            head_dim=head_dim,
            rope_dim=metadata_int(md, "deepseek4.rope.dimension_count", 0) or None,
            rope_base=metadata_float(md, "deepseek4.rope.freq_base", 0.0) or None,
            n_routed_experts=metadata_int(md, "deepseek4.expert_count", 0),
            top_k=metadata_int(md, "deepseek4.expert_used_count", 0),
            expert_intermediate_size=metadata_int(md, "deepseek4.expert_feed_forward_length", 0),
            n_shared_experts=metadata_int(md, "deepseek4.expert_shared_count", 1),
            gate_function="hash+sqrtsoftplus",
            attention_kind="deepseek_v4_mla_compressed_sparse",
            routed_expert_layout="packed_3d",
            norm_eps=metadata_float(md, "deepseek4.attention.layer_norm_rms_epsilon", 0.0) or None,
        )

    def classify_tensor(self, name: str) -> str:
        if "ffn_gate_exps" in name:
            return "routed_w1"
        if "ffn_down_exps" in name:
            return "routed_w2"
        if "ffn_up_exps" in name:
            return "routed_w3"
        if "ffn_gate_shexp" in name or "ffn_up_shexp" in name or "ffn_down_shexp" in name:
            return "shared_experts"
        if "ffn_gate_inp" in name:
            return "gate"
        if "exp_probs_b" in name:
            return "gate_bias"
        if ".attn" in name or "attn_" in name or "indexer" in name:
            return "attention"
        if name == "token_embd.weight":
            return "embedding"
        if name == "output.weight":
            return "lm_head"
        if name == "output_norm.weight":
            return "final_norm"
        if "hc_" in name or name.startswith("output_hc"):
            return "hyper_connection"
        if "ffn" in name:
            return "ffn_other"
        return "other"

    def build_tensor_mappings(self, bundle: GGUFBundle) -> list[TensorMapping]:
        from src.loader.mappings.deepseek_v4 import build_ds4_tensor_mappings

        mappings = []
        for item in build_ds4_tensor_mappings(bundle):
            mappings.append(
                TensorMapping(
                    source_name=item.gguf_name,
                    logical_name=item.target_key,
                    role=item.role or self.classify_tensor(item.gguf_name),
                    transform=item.transform,
                    expert=item.expert,
                )
            )
        return mappings

    def validate_bundle(self, bundle: GGUFBundle) -> SpecValidation:
        params = self.parse_params(bundle)
        errors: list[str] = []
        warnings: list[str] = []
        if bundle.metadata.get("general.architecture") != self.architecture:
            errors.append(f"general.architecture expected {self.architecture!r}, got {bundle.metadata.get('general.architecture')!r}")
        if params.n_layers <= 0:
            errors.append("missing or invalid deepseek4.block_count")
        role_counts = Counter(self.classify_tensor(t.name) for t in bundle.tensors)
        return SpecValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            mapped_sources=sum(role_counts.values()),
            unmapped_sources=[],
            role_counts=dict(role_counts),
        )

    def capability_report(self, bundle: GGUFBundle, *, gpu_count: int = 4, gpu_memory_gib: float = 22.0) -> CapabilityReport:
        params = self.parse_params(bundle)
        role_by_name = {t.name: self.classify_tensor(t.name) for t in bundle.tensors}
        tensor_role_counts = counts_by(bundle.tensors, lambda t: role_by_name[t.name])
        tensor_type_counts = counts_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_type = bytes_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_role = bytes_by(bundle.tensors, lambda t: role_by_name[t.name])
        caps: list[CapabilityItem] = []
        seen: set[tuple[str, str]] = set()
        for t in bundle.tensors:
            role = role_by_name[t.name]
            key = (role, t.type_name)
            if key in seen:
                continue
            seen.add(key)
            status, reason = capability_status_for_role(role, t.type_name, architecture=self.architecture)
            caps.append(CapabilityItem(f"{role}:{t.type_name}", status, reason))
        routed_bytes = sum(int(t.nbytes or 0) for t in bundle.tensors if role_by_name[t.name] in {"routed_w1", "routed_w2", "routed_w3"})
        hardware = HardwareProfile(gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib)
        placements: list[PlacementDecision] = [
            lowbit_device_resident_decision(bundle.size, hardware),
            heterogeneous_expert_decision(routed_bytes),
        ]
        return CapabilityReport(
            architecture=self.architecture,
            params=params,
            tensor_type_counts=tensor_type_counts,
            tensor_role_counts=tensor_role_counts,
            bytes_by_type=bytes_by_type,
            bytes_by_role=bytes_by_role,
            capabilities=caps,
            placements=placements,
        )
