"""Backend-neutral public types for PocketLLM.

These types describe user intent and observable results only.  They deliberately
contain no Torch, CUDA, NCCL, ACL, or backend-specific buffer types.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import ConfigurationError


_BACKENDS = {"auto", "torch", "cpp"}
_FORMATS = {"auto", "safetensors", "gguf"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _coerce_stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        result = tuple(item for item in value if isinstance(item, str) and item)
        if len(result) != len(value):
            raise ConfigurationError("stop must contain only non-empty strings")
        return result
    raise ConfigurationError("stop must be a string, a list of strings, or null")


@dataclass(slots=True)
class EngineArgs:
    """Validated engine construction options shared by both execution planes."""

    model: str = ""
    backend: str = "auto"
    tokenizer_path: str | None = None
    config_path: str | None = None
    model_format: str = "auto"
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    device: str | int | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    kv_cache_dtype: str = "auto"
    prefill_chunk_tokens: int = 0
    enable_prefix_caching: bool = True
    attention_window: int = 0
    attention_sink_tokens: int = 0
    speculative_method: str | None = None
    speculative_tokens: int = 1
    max_batch_size: int = 1
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = str(self.backend).lower()
        self.model_format = str(self.model_format).lower()
        self.kv_cache_dtype = str(self.kv_cache_dtype).lower()
        if self.backend not in _BACKENDS:
            raise ConfigurationError(f"backend must be one of {sorted(_BACKENDS)}, got {self.backend!r}")
        if self.model_format not in _FORMATS:
            raise ConfigurationError(
                f"model_format must be one of {sorted(_FORMATS)}, got {self.model_format!r}"
            )
        if not self.model and not self.backend_options.get("checkpoint_dir"):
            raise ConfigurationError("model/checkpoint path is required")
        if self.tensor_parallel_size < 1:
            raise ConfigurationError("tensor_parallel_size must be >= 1")
        if not 0 <= self.tensor_parallel_rank < self.tensor_parallel_size:
            raise ConfigurationError("tensor_parallel_rank must be in [0, tensor_parallel_size)")
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ConfigurationError("max_model_len must be positive")
        if self.prefill_chunk_tokens < 0:
            raise ConfigurationError("prefill_chunk_tokens must be >= 0")
        if self.attention_window < 0 or self.attention_sink_tokens < 0:
            raise ConfigurationError("attention window and sink tokens must not be negative")
        if self.attention_window == 0 and self.attention_sink_tokens:
            raise ConfigurationError("attention_sink_tokens requires attention_window")
        if self.speculative_tokens < 1:
            raise ConfigurationError("speculative_tokens must be >= 1")
        if self.max_batch_size < 1:
            raise ConfigurationError("max_batch_size must be >= 1")

    @property
    def checkpoint_dir(self) -> str:
        """Return the checkpoint path under either supported spelling."""
        return self.model or str(self.backend_options.get("checkpoint_dir", ""))

    @classmethod
    def from_env(cls, model: str | None = None, **overrides: Any) -> "EngineArgs":
        """Build common options from legacy environment variables.

        This is a compatibility bridge, not the preferred configuration path.
        Backend-specific tuning variables remain available to the selected
        implementation through ``backend_options``.
        """
        values: dict[str, Any] = {
            "model": model or os.getenv("POCKETLLM_MODEL", os.getenv("CKPT_PATH", "")),
            "backend": os.getenv("POCKETLLM_BACKEND", "auto"),
            "tokenizer_path": os.getenv("TOKENIZER_PATH") or None,
            "config_path": os.getenv("CONFIG_PATH") or os.getenv("CONFIG") or None,
            "model_format": os.getenv("CKPT_FORMAT", "auto"),
            "tensor_parallel_size": _env_int("TENSOR_PARALLEL_SIZE", _env_int("TP_WORLD", 1)),
            "tensor_parallel_rank": _env_int("TENSOR_PARALLEL_RANK", _env_int("TP_RANK", 0)),
            "device": os.getenv("DEVICE") or None,
            "max_model_len": (
                _env_int("MAX_MODEL_LEN", _env_int("DEEPSEEK_MAX_MODEL_LEN", 0)) or None
            ),
            "dtype": os.getenv("DTYPE") or None,
            "kv_cache_dtype": os.getenv("POCKETLLM_QWEN_KV_CACHE_DTYPE", os.getenv("KV_CACHE_DTYPE", "auto")),
            "prefill_chunk_tokens": _env_int(
                "PREFILL_CHUNK_TOKENS", _env_int("POCKETLLM_CPP_PREFILL_CHUNK", 0)
            ),
            "enable_prefix_caching": _env_bool("ENABLE_PREFIX_CACHING", True),
            "attention_window": _env_int("QWEN_ATTENTION_WINDOW", 0),
            "attention_sink_tokens": _env_int("QWEN_ATTENTION_SINK_TOKENS", 0),
            "speculative_method": os.getenv("SPECULATIVE_METHOD") or None,
            "speculative_tokens": _env_int("SPECULATIVE_TOKENS", 1),
            "max_batch_size": _env_int(
                "MAX_BATCH_SIZE", _env_int("DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS", 1)
            ),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(slots=True)
class SamplingParams:
    """Per-request sampling and stopping options.

    ``temperature <= 1e-5`` means greedy generation, matching the established
    behavior of both existing runtimes.
    """

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: tuple[str, ...] = ()
    n: int = 1
    logprobs: bool = False
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stop = _coerce_stop(self.stop)
        if self.max_tokens < 1:
            raise ConfigurationError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ConfigurationError("temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ConfigurationError("top_k must be >= 1")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ConfigurationError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ConfigurationError("repetition_penalty must be positive")
        if self.n < 1:
            raise ConfigurationError("n must be >= 1")
        if self.top_logprobs is not None and not 0 <= self.top_logprobs <= 20:
            raise ConfigurationError("top_logprobs must be in [0, 20]")
        if self.logprobs and self.top_logprobs is None:
            self.top_logprobs = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 1.0e-5

    @classmethod
    def from_openai(cls, body: Mapping[str, Any]) -> "SamplingParams":
        """Normalize OpenAI-compatible request fields into one typed object."""
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 256))
        known = {
            "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
            "min_p", "seed", "repetition_penalty", "frequency_penalty",
            "presence_penalty", "stop", "n", "logprobs", "top_logprobs",
            "response_format",
        }
        return cls(
            max_tokens=int(max_tokens),
            temperature=float(body.get("temperature", 0.0) or 0.0),
            top_p=None if body.get("top_p") is None else float(body["top_p"]),
            top_k=None if body.get("top_k") is None else int(body["top_k"]),
            min_p=None if body.get("min_p") is None else float(body["min_p"]),
            seed=None if body.get("seed") is None else int(body["seed"]),
            repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
            frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
            presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
            stop=body.get("stop"),
            n=int(body.get("n", 1) or 1),
            logprobs=bool(body.get("logprobs", False)),
            top_logprobs=None if body.get("top_logprobs") is None else int(body["top_logprobs"]),
            response_format=body.get("response_format"),
            extra={str(k): v for k, v in body.items() if k not in known},
        )

    def to_generation_options(self) -> dict[str, Any]:
        """Return options understood by the current PyTorch generation path."""
        options = {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }
        options.update(self.extra)
        return options


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class TimingMetrics:
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_seconds: float = 0.0
    ttft_seconds: float = 0.0
    tpot_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TimingMetrics":
        values = values or {}
        prefill = float(values.get("prefill_time", values.get("prefill_s", 0.0)) or 0.0)
        decode = float(values.get("decode_time", values.get("decode_s", 0.0)) or 0.0)
        total = float(values.get("total_time", prefill + decode) or (prefill + decode))
        ttft = float(values.get("ttft", prefill) or prefill)
        tpot = float(values.get("tpot", decode) or decode)
        return cls(prefill, decode, total, ttft, tpot)

    def as_dict(self) -> dict[str, float]:
        return {
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "total_seconds": self.total_seconds,
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
        }


@dataclass(slots=True)
class GenerationRequest:
    """A normalized request consumed by a backend."""

    prompt: str | None = None
    prompt_tokens: list[int] | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt is None and self.prompt_tokens is None:
            raise ConfigurationError("either prompt or prompt_tokens is required")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ConfigurationError("prompt must be a string")
        if self.prompt_tokens is not None:
            self.prompt_tokens = [int(token) for token in self.prompt_tokens]
            if not self.prompt_tokens:
                raise ConfigurationError("prompt_tokens must not be empty")
        if not self.request_id:
            raise ConfigurationError("request_id must not be empty")


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    timings: TimingMetrics = field(default_factory=TimingMetrics)
    logprobs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenEvent:
    request_id: str
    token_id: int | None = None
    text: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendCapabilities:
    """Feature declaration used for deterministic routing and API errors."""

    name: str
    models: tuple[str, ...] = ()
    model_formats: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    supports_batch: bool = False
    supports_streaming: bool = True
    supports_cancellation: bool = False
    supports_embeddings: bool = False
    supports_logprobs: bool = False
    supports_structured_outputs: bool = False
    supports_prefix_caching: bool = False
    supports_speculative_decoding: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def supports(self, feature: str) -> bool:
        value = getattr(self, feature, False)
        return bool(value)


@dataclass(slots=True)
class HealthStatus:
    status: str
    backend: str
    ready: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.status not in {"dead", "stopped"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "ready": self.ready,
            "alive": self.alive,
            "message": self.message,
            **self.details,
        }
