"""Optional native C++ backend adapter.

The native module is deliberately optional.  A CUDA/Ascend build can provide
``pocketllm_cpp`` without making the public Python package depend on a vendor
SDK at import time.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    Usage,
    UnsupportedFeatureError,
)
from pocketllm.protocol import encode_chat_prompt, render_fallback_prompt

from .base import BackendBase


_NATIVE_MODULE_NAMES = ("pocketllm_cpp", "_pocketllm_cpp", "cpp_engine")


def _preload_torch_runtime() -> None:
    """Let torch bind its own NCCL before the native module loads one.

    The extension links its own libnccl.  If it is imported first, that copy
    wins the global symbol lookup and a newer ``libtorch_cuda.so`` can then fail
    on a symbol it does not provide (observed: ``undefined symbol:
    ncclCommResume``).  Importing torch first is enough to fix the order.  Torch
    is optional for this backend, so a missing or broken install is ignored here
    and reported later by whatever actually needs it.
    """
    try:
        importlib.import_module("torch")
    except Exception:
        return


def load_native_module() -> Any:
    _preload_torch_runtime()
    errors: list[str] = []
    for name in _NATIVE_MODULE_NAMES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        # The repository's ``cpp_engine`` directory is importable as a Python
        # namespace package even when no extension was built.  Treat that as
        # unavailable instead of returning a module that cannot construct an
        # engine, which would also make ``backend=auto`` select C++ incorrectly.
        if not hasattr(module, "QwenEngine"):
            errors.append(f"{name}: QwenEngine binding is missing")
            continue
        return module
    raise BackendUnavailableError(
        "native C++ backend is unavailable; build cpp_engine with "
        "-DPOCKET_BUILD_PYTHON=ON (" + "; ".join(errors) + ")"
    )


def _native_kv_cache_dtype(value: str) -> str:
    """Resolve the public ``auto`` value to the native Qwen default."""
    normalized = str(value or "auto").lower()
    return "fp16" if normalized == "auto" else normalized


def _coerce_eos_ids(value: Any, *, source: str) -> tuple[int, ...]:
    """Normalize a configured EOS value into a tuple of token ids."""
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ConfigurationError(f"{source} eos_token_id must be an integer or list of integers")
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigurationError(f"{source} eos_token_id list must contain integers only")
            ids.append(int(item))
        return tuple(dict.fromkeys(ids))
    raise ConfigurationError(f"{source} eos_token_id must be an integer or list of integers")


def _checkpoint_eos_ids(checkpoint_dir: str) -> tuple[tuple[int, ...], str]:
    """Read the EOS ids a checkpoint declares, preferring generation_config.json."""
    if not checkpoint_dir:
        return (), ""
    for name in ("generation_config.json", "config.json"):
        path = os.path.join(checkpoint_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        ids = _coerce_eos_ids(data.get("eos_token_id"), source=name)
        if ids:
            return ids, name
    return (), ""


def _native_device_index(value: str | int | None) -> int:
    """Normalize a public device selector for the native single-rank option."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ConfigurationError("C++ backend device must be a non-negative device index")
    if isinstance(value, int):
        index = value
    else:
        text = str(value).strip().lower()
        for prefix in ("cuda:", "ascend:", "npu:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        try:
            index = int(text)
        except ValueError as exc:
            raise ConfigurationError(
                "C++ backend device must be an integer or cuda:/ascend:/npu: index"
            ) from exc
    if index < 0:
        raise ConfigurationError("C++ backend device must be a non-negative device index")
    return index


class CppBackend(BackendBase):
    """Serialized compatibility adapter for the stateful native engines.

    The initial bridge intentionally supports Qwen's token-oriented API.  The
    native engine still owns its optimized KV layout and TP protocol; this
    adapter does not turn it into a Torch module or copy its buffers.
    """

    def __init__(
        self,
        args: EngineArgs,
        *,
        native_module: Any | None = None,
        engine: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        if native_module is not None:
            self._native = native_module
        elif engine is None:
            self._native = load_native_module()
        else:
            self._native = None
        self._tokenizer_error: str | None = None
        self._engine = engine if engine is not None else self._construct_engine()
        # A primitive lock may be released by a different worker thread.  This
        # matters for AsyncLLM, which advances one generator through an executor
        # without guaranteeing that every next() call uses the same thread.
        self._request_lock = threading.Lock()
        self._tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._eos_ids, self._eos_source = self._resolve_eos_ids()

        # Phase 3.4: Optional batch scheduler.  Only rank 0 drives scheduling:
        # the scheduler runs a background thread that issues collectives, and on
        # a worker rank that would race run_worker_loop() and deadlock NCCL.
        self._scheduler = None
        self._batching_enabled = False
        if (
            self.args.backend_options.get("enable_batching", False)
            and self.args.tensor_parallel_rank == 0
        ):
            self._batching_enabled = self._init_batch_scheduler()

        self._ready = True

    def _resolve_eos_ids(self) -> tuple[frozenset[int], str]:
        """Resolve the stop tokens without guessing a vocabulary-specific id.

        ``backend_options["eos_token_id"]`` wins so an operator can override a
        checkpoint.  Otherwise the native engine, the tokenizer, and finally the
        checkpoint's ``generation_config.json``/``config.json`` are consulted.
        An empty result means only the token budget can end generation, which the
        adapter reports as ``finish_reason="length"``.
        """
        override = _coerce_eos_ids(self.args.backend_options.get("eos_token_id"), source="backend_options")
        if override:
            return frozenset(override), "backend_options"
        for holder, attribute, source in (
            (self._engine, "eos_id", "native engine"),
            (self._engine, "eos_token_id", "native engine"),
        ):
            if holder is None:
                continue
            value = getattr(holder, attribute, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            ids = _coerce_eos_ids(value, source=source)
            if ids:
                return frozenset(ids), source
        config = getattr(self._engine, "config", None)
        if isinstance(config, dict):
            ids = _coerce_eos_ids(config.get("eos_token_id"), source="native config")
            if ids:
                return frozenset(ids), "native config"
        # generation_config.json is consulted before the tokenizer because it is
        # what the checkpoint declares for generation. Chat checkpoints commonly
        # stop on a turn-end token that differs from the tokenizer's EOS.
        ids, name = _checkpoint_eos_ids(self.args.checkpoint_dir)
        if ids:
            return frozenset(ids), name
        value = getattr(self._tokenizer, "eos_token_id", None) if self._tokenizer is not None else None
        ids = _coerce_eos_ids(value, source="tokenizer")
        if ids:
            return frozenset(ids), "tokenizer"
        return frozenset(), ""

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_ids

    def _configured_max_batch_size(self) -> int:
        """Batch width for this backend, shared by the engine and the scheduler.

        The engine sizes its KV cache from this at construction, so the scheduler
        must not ask for more slots later.  Batching off means a single slot,
        which keeps the serial path's memory footprint unchanged.
        """
        if not self.args.backend_options.get("enable_batching", False):
            return 1
        try:
            requested = int(self.args.backend_options.get("max_batch_size", 8))
        except (TypeError, ValueError):
            return 1
        return requested if requested >= 1 else 1

    def _init_batch_scheduler(self) -> bool:
        """Initialize the batch scheduler if available."""
        if not hasattr(self._native, "QwenBatchScheduler"):
            import warnings
            warnings.warn(
                "enable_batching=True but native module does not expose QwenBatchScheduler; "
                "falling back to serial execution"
            )
            return False

        max_batch_size = self._configured_max_batch_size()
        if max_batch_size <= 1:
            return False

        try:
            self._scheduler = self._native.QwenBatchScheduler(self._engine, max_batch_size)
            return True
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to create QwenBatchScheduler: {e}; falling back to serial execution")
            return False

    @staticmethod
    def native_available() -> bool:
        try:
            load_native_module()
            return True
        except BackendUnavailableError:
            return False

    def _registered_architectures(self) -> tuple[str, ...]:
        """Models this build can serve, read from the native engine registry.

        This is the same list the C++ CLI dispatches on, so a build that links a
        second engine reports it here without anyone editing a literal.  The
        fallback covers a native module predating the registry, and a
        capabilities report is not worth failing a backend over.
        """
        listed = getattr(self._native, "registered_architectures", None)
        if listed is None:
            return ("qwen3_5",)
        try:
            return tuple(str(name) for name in listed())
        except Exception:
            return ("qwen3_5",)

    @property
    def capabilities(self) -> BackendCapabilities:
        # The Ascend build rejects the external drafters, so report only what the
        # linked backend actually implements instead of the CUDA superset.
        native_backend = str(getattr(self._native, "backend", "") or "").lower()
        speculative: tuple[str, ...] = ("mtp",) if native_backend == "ascend" else ("mtp", "dspark", "dflash2")
        return BackendCapabilities(
            name="cpp",
            models=self._registered_architectures(),
            model_formats=("safetensors",),
            devices=(native_backend,) if native_backend else ("cuda", "ascend"),
            supports_batch=self._batching_enabled,  # Phase 3.4: dynamic based on scheduler
            supports_streaming=True,
            supports_cancellation=True,
            supports_embeddings=False,
            supports_logprobs=False,
            supports_structured_outputs=False,
            supports_prefix_caching=True,
            supports_speculative_decoding=speculative,
            details={
                "execution": "native C++",
                "scheduler": "batch scheduler" if self._batching_enabled else "serialized compatibility session",
                "device_backend": native_backend or "unknown",
                "cancellation": "safe boundary only",
                "eos_token_ids": sorted(self._eos_ids),
                "eos_source": self._eos_source or "none",
                "max_batch_size": self._configured_max_batch_size() if self._batching_enabled else 1,
                "kv_paged": bool(self.args.backend_options.get("kv_paged", False)),
            },
        )

    def _load_tokenizer(self) -> Any | None:
        tokenizer_path = self.args.tokenizer_path or self.args.checkpoint_dir
        if not tokenizer_path:
            self._tokenizer_error = "no tokenizer_path and no checkpoint_dir"
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as exc:
            # Keep the reason. A token-only caller still works without a
            # tokenizer, so this is not fatal here, but swallowing it turns a
            # broken transformers/torch install into a misleading "needs
            # tokenizer_path" error at the first text prompt.
            self._tokenizer_error = f"{type(exc).__name__}: {exc}"
            return None

    def _construct_engine(self) -> Any:
        # Which engine opens a checkpoint is the checkpoint's own declaration,
        # read through the same C++ model registry the native CLI dispatches on.
        # `engine_kind` remains as an explicit override for the cases detection
        # cannot cover -- a directory without a config.json, or forcing one
        # runtime onto a checkpoint for a comparison.
        kind = str(self.args.backend_options.get("engine_kind", "auto")).lower()
        if kind == "auto":
            kind = self._detect_engine_kind()

        if kind == "persistent":
            return self._construct_persistent_engine()
        elif kind == "qwen":
            return self._construct_qwen_engine()
        else:
            raise UnsupportedFeatureError(
                f"unsupported engine_kind: {kind}; use 'auto', 'persistent' or 'qwen'"
            )

    # Architectures the registry knows, mapped to the native engine class this
    # backend constructs for them.  The backend builds the concrete classes
    # rather than going through create_engine() because it needs their full
    # option surface (KV dtype, drafters, prefill chunk), which the registry's
    # model-agnostic EngineOptions deliberately does not carry.
    _ENGINE_KIND_BY_ARCHITECTURE = {
        "qwen3_5": "qwen",
        "deepseek_v4": "persistent",
    }

    def _detect_engine_kind(self) -> str:
        """Ask the native registry which engine this checkpoint wants."""
        detect = getattr(self._native, "detect_architecture", None)
        checkpoint = self.args.checkpoint_dir
        if detect is None or not checkpoint:
            # An older native module, or a caller that passed no checkpoint at
            # all (token-only tests).  Keep the previous default rather than
            # failing on a path that never needed detecting.
            return "qwen"
        try:
            architecture = str(detect(checkpoint))
        except Exception as exc:
            raise ConfigurationError(
                f"could not detect the model architecture of {checkpoint}: {exc}; "
                "pass --engine-kind qwen or --engine-kind persistent to choose one"
            ) from exc

        kind = self._ENGINE_KIND_BY_ARCHITECTURE.get(architecture)
        if kind is None:
            known = ", ".join(sorted(self._ENGINE_KIND_BY_ARCHITECTURE)) or "none"
            raise UnsupportedFeatureError(
                f"no cpp backend engine for architecture "
                f"'{architecture or '<undeclared>'}' declared by {checkpoint}; "
                f"known architectures: {known}"
            )
        return kind

    def _construct_persistent_engine(self) -> Any:
        """Construct PersistentEngine (supports TP worker loop)."""
        cls = getattr(self._native, "PersistentEngine", None)
        options_cls = getattr(self._native, "ForwardSmokeOptions", None)
        if cls is None or options_cls is None:
            raise BackendUnavailableError("native module does not expose PersistentEngine bindings")

        options = options_cls()
        options.tp_world = self.args.tensor_parallel_size
        options.tp_rank = self.args.tensor_parallel_rank
        options.device = _native_device_index(self.args.device)
        options.skip_fp4_host_prepare = False
        options.nccl_id_path = str(self.args.backend_options.get("nccl_id_path", ""))

        layer_count = 0  # auto-detect from checkpoint
        max_context = self.args.max_model_len or 8192

        return cls(self.args.checkpoint_dir, options, layer_count, max_context)

    def _construct_qwen_engine(self) -> Any:
        """Construct QwenEngine, which drives TP through its own worker loop."""
        cls = getattr(self._native, "QwenEngine", None)
        options_cls = getattr(self._native, "QwenEngineOptions", None)
        if cls is None or options_cls is None:
            raise BackendUnavailableError("native module does not expose QwenEngine bindings")

        # Get NCCL ID path from backend_options or environment
        nccl_id_path = str(self.args.backend_options.get("nccl_id_path", ""))
        if not nccl_id_path:
            nccl_id_path = os.environ.get("POCKETLLM_NCCL_ID_PATH", "")
        options = options_cls()
        mappings = {
            "tp_world": self.args.tensor_parallel_size,
            "tp_rank": self.args.tensor_parallel_rank,
            "device": self._native_rank_device(),
            # The KV cache is sized at construction, so the batch width must be
            # known here; the scheduler cannot grow it afterwards.
            "max_batch_size": self._configured_max_batch_size(),
            "prefill_chunk_tokens": self.args.prefill_chunk_tokens or 8192,
            "attention_window": self.args.attention_window,
            "attention_sink_tokens": self.args.attention_sink_tokens,
            "prefix_cache": self.args.enable_prefix_caching,
            "state_snapshot_interval_tokens": int(self.args.backend_options.get("state_snapshot_interval_tokens", 4096)),
            "max_state_snapshots": int(self.args.backend_options.get("max_state_snapshots", 82)),
            "mtp": self.args.speculative_method == "mtp",
            "mtp_speculative_tokens": self.args.speculative_tokens,
            "mtp_adaptive": bool(self.args.backend_options.get("mtp_adaptive", False)),
            "dspark_checkpoint": str(self.args.backend_options.get("dspark_checkpoint", "")),
            "dflash2_checkpoint": str(self.args.backend_options.get("dflash2_checkpoint", "")),
            "nccl_id_path": nccl_id_path,
            # Paged KV (Phase 3.7). Off by default; the contiguous arena
            # reproduces the max_batch_size * max_context reservation. A zero
            # kv_cache_bytes derives the pool from that same reservation, so
            # turning paging on alone is memory-neutral until the budget is
            # raised deliberately.
            "kv_paged": bool(self.args.backend_options.get("kv_paged", False)),
            "kv_block_size": int(self.args.backend_options.get("kv_block_size", 16)),
            "kv_cache_bytes": int(self.args.backend_options.get("kv_cache_bytes", 0)),
        }
        for name, value in mappings.items():
            if hasattr(options, name):
                setattr(options, name, value)
        if hasattr(options, "kv_cache_dtype") and hasattr(self._native, "parse_qwen_kv_cache_dtype"):
            setattr(
                options,
                "kv_cache_dtype",
                self._native.parse_qwen_kv_cache_dtype(_native_kv_cache_dtype(self.args.kv_cache_dtype)),
            )
        for name, value in {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "sampling_seed": 0,
        }.items():
            if hasattr(options, name):
                setattr(options, name, value)
        engine = cls(self.args.checkpoint_dir, options, 0, self.args.max_model_len or 8192)
        # warmup_tp() builds the command channel and forces the NCCL
        # communicator up.  Both ranks must do it before any rank issues a
        # collective, and a worker cannot enter run_worker_loop() without it.
        if self.args.tensor_parallel_size > 1:
            engine.warmup_tp()
        return engine

    def _native_rank_device(self) -> int:
        """Resolve this rank's device index.

        An explicit --device wins.  Otherwise a TP rank claims device
        ``tensor_parallel_rank``, since the supervisor gives every rank the same
        visible device list; leaving the default 0 would stack the whole world
        onto one GPU.  When the launcher already narrowed CUDA_VISIBLE_DEVICES to
        one device per rank, that device is renumbered to 0 for this process, so
        the rank offset must not be applied on top of it.
        """
        if self.args.device is not None:
            return _native_device_index(self.args.device)
        if self.args.tensor_parallel_size <= 1:
            return 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            entries = [item for item in visible.split(",") if item.strip()]
            if len(entries) <= 1:
                return 0
        return self.args.tensor_parallel_rank

    def health(self) -> HealthStatus:
        status = super().health()
        details = dict(status.details)
        details.update({"model": self.args.checkpoint_dir})
        return HealthStatus(status.status, status.backend, status.ready, status.message, details)

    def _prompt_ids(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        if self._tokenizer is None:
            detail = f" (tokenizer load failed: {self._tokenizer_error})" if self._tokenizer_error else ""
            raise ConfigurationError(
                f"C++ backend needs a working tokenizer for text prompts{detail}"
            )
        messages = request.metadata.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            encoded = encode_chat_prompt(
                self._tokenizer,
                messages,
                thinking_mode=str(request.metadata.get("thinking_mode", "chat")),
                reasoning_effort=request.metadata.get("reasoning_effort"),
                tools=request.metadata.get("tools"),
            )
            if encoded is not None:
                return encoded
            prompt = request.prompt or render_fallback_prompt(messages)
        else:
            prompt = request.prompt or ""
        encoded = self._tokenizer.encode(prompt)
        return [int(token) for token in encoded]

    def _check_sampling(self, params: SamplingParams) -> None:
        if not params.greedy:
            raise UnsupportedFeatureError("the initial C++ adapter exposes native greedy generation only")
        if params.top_p is not None or params.top_k is not None or params.min_p is not None:
            raise UnsupportedFeatureError("C++ sampling controls are not exposed by the initial binding")
        if params.n != 1 or params.logprobs or params.stop:
            raise UnsupportedFeatureError("n, logprobs, and stop require the shared scheduler phase")

    @staticmethod
    def _native_token(item: Any) -> int:
        if isinstance(item, int):
            return int(item)
        return int(getattr(item, "top_token", getattr(item, "token", item)))

    # --- TP command fan-out -------------------------------------------------
    # Under TP the worker ranks sit in run_worker_loop() waiting for a command.
    # Every rank has to enter the same collective in the same order, so rank 0
    # announces the op before it runs the op itself.  At world size 1 the
    # worker_command_* calls are native no-ops, so these wrappers stay on the
    # single-rank path unchanged.

    def _engine_or_raise(self) -> Any:
        engine = self._engine
        if engine is None:
            raise RuntimeError("native C++ engine is closed")
        return engine

    def _tp_command(self, name: str, *args: Any) -> None:
        if self.args.tensor_parallel_size <= 1:
            return
        engine = self._engine_or_raise()
        command = getattr(engine, name, None)
        if command is None:
            raise UnsupportedFeatureError(
                f"native engine does not expose {name}; "
                "rebuild cpp_engine with -DPOCKET_BUILD_PYTHON=ON"
            )
        command(*args)

    def _tp_prefill(self, prompt_ids: list[int]) -> Any:
        self._tp_command("worker_command_prefill", prompt_ids)
        return self._engine_or_raise().prefill(prompt_ids)

    def _tp_decode_step(self, token: int) -> Any:
        self._tp_command("worker_command_decode", token)
        return self._engine_or_raise().decode_step(token)

    def _is_eos(self, token: int) -> bool:
        return token in self._eos_ids

    def _generate_until_eos(
        self, request: GenerationRequest, prompt_ids: list[int], started: float
    ) -> tuple[list[int], bool, float]:
        """Step the native session so it never advances past EOS.

        ``QwenEngine.generate`` takes no EOS argument and keeps mutating its
        session for the whole token budget.  Calling it and truncating the result
        would leave the recurrent state and prefix cache positioned after text
        the caller never sees, corrupting reuse for the next request.  Driving
        prefill/decode here keeps the session and the returned tokens in step,
        and matches what the streamed path does.

        Returns the tokens, whether EOS stopped generation, and the time to the
        first token measured from ``started``.  TTFT is observable only here: the
        loop is what learns when prefill produced a token, so reporting 0.0 as
        this path used to made every serial timing comparison degenerate.
        """
        result = self._tp_prefill(prompt_ids)
        ttft = time.perf_counter() - started
        token_ids: list[int] = []
        for index in range(request.sampling_params.max_tokens):
            self._ensure_open()
            self._check_cancelled(request.request_id)
            token = self._native_token(result)
            if self._is_eos(token):
                return token_ids, True, ttft
            token_ids.append(token)
            if index + 1 < request.sampling_params.max_tokens:
                result = self._tp_decode_step(token)
        return token_ids, False, ttft

    def _decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            return ""
        decode = getattr(self._tokenizer, "decode", None)
        if callable(decode):
            return str(decode(token_ids))
        decode_tokens = getattr(self._tokenizer, "decode_tokens", None)
        if callable(decode_tokens):
            return str(decode_tokens(token_ids))
        return ""

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests with optional batch scheduler."""
        self._ensure_open()

        # Phase 3.4: Use batch scheduler if enabled
        if self._batching_enabled and self._scheduler is not None:
            return self._generate_batched(requests)

        # Legacy serial path
        return self._generate_serial(requests)

    def _generate_serial(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests serially (legacy path)."""
        outputs: list[GenerationResult] = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_sampling(request.sampling_params)
                self._check_cancelled(request.request_id)
                prompt_ids = self._prompt_ids(request)
                started = time.perf_counter()
                with self._request_lock:
                    self._ensure_open()
                    self._check_cancelled(request.request_id)
                    # Native generate() drives its own prefill/decode loop
                    # internally, so rank 0 cannot announce those steps to the
                    # workers. Under TP always take the stepped path, which
                    # broadcasts each command.
                    if self._eos_ids or self.args.tensor_parallel_size > 1:
                        token_ids, hit_eos, ttft = self._generate_until_eos(
                            request, prompt_ids, started
                        )
                    else:
                        raw = self._engine_or_raise().generate(
                            prompt_ids, request.sampling_params.max_tokens
                        )
                        token_ids = [self._native_token(item) for item in raw]
                        hit_eos = False
                        # Native generate() drives its own loop and reports no
                        # per-token boundary, so the first token is not separable
                        # from the whole call here.
                        ttft = 0.0
                    self._ensure_open()
                self._check_cancelled(request.request_id)
                # `completion_tokens` counts the EOS step the engine executed, so
                # usage stays comparable with the streamed path.
                completion_tokens = len(token_ids) + (1 if hit_eos else 0)
                outputs.append(
                    GenerationResult(
                        request_id=request.request_id,
                        token_ids=token_ids,
                        text=self._decode(token_ids),
                        finish_reason="stop" if hit_eos else "length",
                        usage=Usage(len(prompt_ids), completion_tokens),
                        timings=TimingMetrics(
                            total_seconds=time.perf_counter() - started,
                            ttft_seconds=ttft,
                        ),
                    )
                )
            finally:
                self._clear_request(request.request_id)
                if self._closed:
                    self._release_native()
        return outputs

    def _generate_batched(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests using the batch scheduler."""
        if not requests:
            return []

        # Submit all requests to scheduler
        request_map: dict[int, GenerationRequest] = {}
        native_request_ids: list[int] = []

        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_sampling(request.sampling_params)
                prompt_ids = self._prompt_ids(request)

                # Create native sampling params
                sampling = self._native.QwenBatchSamplingParams()
                sampling.max_new_tokens = request.sampling_params.max_tokens
                sampling.temperature = request.sampling_params.temperature or 0.0
                sampling.top_p = request.sampling_params.top_p or 1.0
                sampling.top_k = request.sampling_params.top_k or 20
                # The batch scheduler runs the model to max_new_tokens unless
                # told otherwise. EOS truncation is the default for serving, but
                # a benchmark (or any caller that wants the full budget) opts in
                # through extra["ignore_eos"], mirroring the native parity driver
                # which sets req.sampling.ignore_eos = true.
                if bool(request.sampling_params.extra.get("ignore_eos", False)):
                    sampling.ignore_eos = True

                # Submit to scheduler
                native_req_id = self._scheduler.submit_request(prompt_ids, sampling, None)
                if native_req_id > 0:
                    native_request_ids.append(native_req_id)
                    request_map[native_req_id] = request
                else:
                    # Submission failed
                    self._clear_request(request.request_id)

            except Exception as e:
                self._clear_request(request.request_id)
                raise

        # Poll for results
        outputs: list[GenerationResult] = []
        timeout_ms = 60000  # 60 seconds per request

        for native_req_id in native_request_ids:
            request = request_map[native_req_id]
            result = self._scheduler.poll_result(native_req_id, timeout_ms)

            if result is None:
                # Timeout
                self._clear_request(request.request_id)
                raise TimeoutError(f"Request {request.request_id} timed out after {timeout_ms}ms")

            # Convert native result to GenerationResult
            token_ids = result.generated_tokens
            text = self._decode(token_ids)

            outputs.append(
                GenerationResult(
                    request_id=request.request_id,
                    token_ids=token_ids,
                    text=text,
                    finish_reason=result.finish_reason,
                    usage=Usage(result.prompt_tokens, result.completion_tokens),
                    timings=TimingMetrics(
                        total_seconds=result.total_seconds,
                        ttft_seconds=result.ttft_seconds,
                    ),
                )
            )

            self._clear_request(request.request_id)

        return outputs

    def _stream_native(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self._check_sampling(request.sampling_params)
        prompt_ids = self._prompt_ids(request)
        max_tokens = request.sampling_params.max_tokens
        self._engine_or_raise()
        # Do not reset() here.  QwenEngine::reset() clears the prefix cache, so
        # calling it per request would disable configured prefix reuse.  prefill()
        # already matches the common prefix, restores a snapshot, or zeroes the
        # recurrent state, which is also what native generate() relies on.
        # QwenEngine.prefill predicts the first token without consuming it. The
        # following decode steps consume the previous prediction and predict the
        # next one, matching the native generate() result ordering.
        result = self._tp_prefill(prompt_ids)
        generated: list[int] = []
        previous_text = ""
        for index in range(max_tokens):
            self._ensure_open()
            self._check_cancelled(request.request_id)
            token = self._native_token(result)
            if self._is_eos(token):
                # Stop before another decode_step. EOS is counted as a generated
                # step but is not emitted as visible text.
                yield TokenEvent(
                    request.request_id,
                    finish_reason="stop",
                    usage=Usage(len(prompt_ids), len(generated) + 1),
                )
                return
            generated.append(token)
            decoded = self._decode(generated)
            # Decode the complete sequence so BPE/UTF-8 token boundaries are handled
            # by the tokenizer. Emit only the newly visible suffix when possible.
            text = decoded[len(previous_text):] if decoded.startswith(previous_text) else decoded
            previous_text = decoded
            event = TokenEvent(request.request_id, token_id=token, text=text)
            if index + 1 == max_tokens:
                event.finish_reason = "length"
                event.usage = Usage(len(prompt_ids), len(generated))
            yield event
            if index + 1 < max_tokens:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                result = self._tp_decode_step(token)

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        # A primitive lock can span generator yields even when AsyncLLM resumes
        # the generator on a different executor thread. It serializes all access
        # to the native engine's single mutable KV-cache session.
        self._begin_request(request.request_id)
        with self._request_lock:
            try:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                yield from self._stream_native(request)
            finally:
                self._clear_request(request.request_id)
                if self._closed:
                    self._release_native()

    def _release_native(self) -> None:
        engine = self._engine
        # Workers block in read() on the command channel. Without a shutdown
        # they never return from run_worker_loop() and the supervisor has to
        # escalate to SIGKILL.
        if (
            engine is not None
            and self.args.tensor_parallel_size > 1
            and self.args.tensor_parallel_rank == 0
        ):
            shutdown = getattr(engine, "worker_command_shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    # The workers may already be gone; closing is best effort.
                    pass
        self._engine = None
        self._tokenizer = None
        self._native = None
        self._ready = False
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Enter the native worker loop for a nonzero TP rank.

        This blocks until rank 0 sends a shutdown command through the command
        channel. The engine must already be constructed.
        """
        self._ensure_open()
        if self.args.tensor_parallel_rank == 0:
            raise RuntimeError("run_worker must not be called on rank 0")
        if self._engine is None:
            raise RuntimeError("native engine is not constructed")
        if not hasattr(self._engine, "run_worker_loop"):
            raise UnsupportedFeatureError(
                "native engine does not expose run_worker_loop; "
                "rebuild cpp_engine with -DPOCKET_BUILD_PYTHON=ON"
            )
        if on_ready is not None:
            on_ready()
        # run_worker_loop() blocks until rank 0 sends shutdown
        self._engine.run_worker_loop()

    def close(self) -> None:
        if self._closed:
            return
        super().close()
        # Never destroy a native engine while a GIL-released kernel is using it.
        # An active generate/stream call releases it from its own finally block.
        if self._request_lock.acquire(blocking=False):
            try:
                self._release_native()
            finally:
                self._request_lock.release()

        # Clean up supervisor if this backend owns it
        if hasattr(self, "_supervisor"):
            try:
                self._supervisor.stop()
            except Exception:
                pass
