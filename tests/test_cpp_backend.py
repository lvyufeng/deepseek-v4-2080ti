from __future__ import annotations

import threading
import time
import pytest

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    RequestCancelledError,
    SamplingParams,
    UnsupportedFeatureError,
)
from pocketllm.backends.cpp_backend import CppBackend
from pocketllm.cli import _args, build_parser


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text), 7]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class TemplateTokenizer(FakeTokenizer):
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
        return [71, 72]



class FakeResult:
    def __init__(self, top_token: int) -> None:
        self.top_token = top_token


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0
        self.active = 0
        self.max_active = 0
        self._active_lock = threading.Lock()

    def generate(self, prompt_ids: list[int], max_tokens: int) -> list[FakeResult]:
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(("generate", list(prompt_ids), max_tokens))
            time.sleep(0.02)
            return [FakeResult(10 + index) for index in range(max_tokens)]
        finally:
            with self._active_lock:
                self.active -= 1

    def reset(self) -> None:
        self.calls.append(("reset",))

    def clear_prefix_cache(self) -> None:
        self.calls.append(("clear_prefix_cache",))

    def prefill(self, prompt_ids: list[int]) -> FakeResult:
        self.calls.append(("prefill", list(prompt_ids)))
        return FakeResult(10)

    def decode_step(self, token: int) -> FakeResult:
        self.calls.append(("decode_step", token))
        return FakeResult(token + 1)

    def close(self) -> None:
        self.closed += 1


class FakeOptions:
    def __init__(self) -> None:
        self.tp_world = 1
        self.tp_rank = 0
        self.device = 0
        self.prefill_chunk_tokens = 0
        self.kv_cache_dtype = "unset"
        self.attention_window = 0
        self.attention_sink_tokens = 0
        self.prefix_cache = False
        self.state_snapshot_interval_tokens = 0
        self.max_state_snapshots = 0
        self.mtp = False
        self.mtp_speculative_tokens = 1
        self.mtp_adaptive = False
        self.dspark_checkpoint = ""
        self.dflash2_checkpoint = ""
        self.nccl_id_path = ""
        self.temperature = 1.0
        self.top_p = 0.0
        self.top_k = 0
        self.sampling_seed = -1


class FakeNativeModule:
    QwenEngineOptions = FakeOptions

    def __init__(self) -> None:
        self.constructed: tuple[str, FakeOptions, int, int] | None = None
        self.detected: list[str] = []

    @staticmethod
    def parse_qwen_kv_cache_dtype(value: str) -> str:
        return f"native:{value}"

    @staticmethod
    def registered_architectures() -> list[str]:
        return ["deepseek_v4", "qwen3_5"]

    def detect_architecture(self, checkpoint: str) -> str:
        self.detected.append(checkpoint)
        return "qwen3_5"

    def QwenEngine(
        self,
        checkpoint: str,
        options: FakeOptions,
        layer_count: int,
        max_context: int,
    ) -> FakeEngine:
        self.constructed = (checkpoint, options, layer_count, max_context)
        return FakeEngine()


def make_backend(engine: FakeEngine | None = None) -> tuple[CppBackend, FakeEngine]:
    fake_engine = engine or FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=fake_engine,
        tokenizer=FakeTokenizer(),
    )
    return backend, fake_engine


def test_cpp_chat_prompt_uses_tokenizer_owned_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-template",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_mode": "chat",
        },
    )

    assert backend._prompt_ids(request) == [71, 72]
    assert tokenizer.template_calls[0][0] == [{"role": "user", "content": "hi"}]
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    backend.close()


def test_cpp_pre_tokenized_prompt_bypasses_chat_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt_tokens=[7, 8],
        request_id="req-template-tokens",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": [{"role": "user", "content": "ignored"}]},
    )

    assert backend._prompt_ids(request) == [7, 8]
    assert tokenizer.template_calls == []
    backend.close()


def test_cpp_raw_prompt_bypasses_chat_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt="raw completion",
        request_id="req-template-raw",
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert backend._prompt_ids(request) == [len("raw completion"), 7]
    assert tokenizer.template_calls == []
    backend.close()


def test_generate_converts_native_results_and_usage() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt="hello",
        request_id="req-generate",
        sampling_params=SamplingParams(max_tokens=3),
    )

    result = backend.generate([request])[0]

    # Without a known EOS the adapter delegates to native generate().
    assert engine.calls == [("generate", [5, 7], 3)]
    assert result.request_id == "req-generate"
    assert result.token_ids == [10, 11, 12]
    assert result.text == "<10><11><12>"
    assert result.finish_reason == "length"
    assert result.usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


def test_stream_matches_native_prefill_decode_order() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=3),
    )

    events = list(backend.stream(request))

    # No reset()/clear_prefix_cache(): native prefill() owns prefix matching, so
    # resetting per request would disable configured prefix reuse.
    assert engine.calls == [
        ("prefill", [4, 5]),
        ("decode_step", 10),
        ("decode_step", 11),
    ]
    assert [event.token_id for event in events] == [10, 11, 12]
    assert [event.text for event in events] == ["<10>", "<11>", "<12>"]
    assert [event.finish_reason for event in events] == [None, None, "length"]
    assert events[-1].usage is not None
    assert events[-1].usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


def test_stream_cancellation_stops_before_next_decode_step() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt_tokens=[4],
        request_id="req-cancel",
        sampling_params=SamplingParams(max_tokens=3),
    )
    stream = backend.stream(request)

    assert next(stream).token_id == 10
    assert backend.cancel(request.request_id) is True
    with pytest.raises(RequestCancelledError):
        next(stream)

    assert engine.calls == [("prefill", [4])]


def test_requests_are_serialized_around_one_native_session() -> None:
    backend, engine = make_backend()
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def generate(index: int) -> None:
        try:
            barrier.wait()
            backend.generate(
                [
                    GenerationRequest(
                        prompt_tokens=[index + 1],
                        request_id=f"req-{index}",
                        sampling_params=SamplingParams(max_tokens=1),
                    )
                ]
            )
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=generate, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2.0)

    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert engine.max_active == 1


def test_close_releases_native_engine_and_rejects_new_work() -> None:
    backend, engine = make_backend()

    backend.close()
    backend.close()

    assert engine.closed == 1
    assert backend.health().status == "stopped"
    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.generate(
            [GenerationRequest(prompt_tokens=[1], request_id="req-closed")]
        )


def test_close_during_stream_releases_at_safe_boundary() -> None:
    backend, engine = make_backend()
    stream = backend.stream(
        GenerationRequest(
            prompt_tokens=[1],
            request_id="req-close-stream",
            sampling_params=SamplingParams(max_tokens=2),
        )
    )

    assert next(stream).token_id == 10
    backend.close()
    assert engine.closed == 0
    with pytest.raises(RuntimeError, match="backend is closed"):
        next(stream)
    assert engine.closed == 1


def test_native_options_normalize_cli_device_and_auto_kv_dtype() -> None:
    namespace = build_parser().parse_args(
        [
            "serve",
            "--model",
            "checkpoint",
            "--backend",
            "cpp",
            "--device",
            "cuda:2",
            "--max-model-len",
            "4096",
        ]
    )
    native = FakeNativeModule()

    backend = CppBackend(_args(namespace), native_module=native, tokenizer=FakeTokenizer())

    assert native.constructed is not None
    checkpoint, options, layer_count, max_context = native.constructed
    assert checkpoint == "checkpoint"
    assert options.device == 2
    assert options.kv_cache_dtype == "native:fp16"
    assert options.temperature == 0.0
    assert (layer_count, max_context) == (0, 4096)
    backend.close()


@pytest.mark.parametrize("device", ["gpu", "cuda:", -1, True])
def test_invalid_native_device_is_rejected(device: object) -> None:
    native = FakeNativeModule()
    with pytest.raises(ConfigurationError, match="device"):
        CppBackend(
            EngineArgs(model="checkpoint", backend="cpp", device=device),
            native_module=native,
            tokenizer=FakeTokenizer(),
        )


def test_cpp_capabilities_match_phase_one_surface() -> None:
    backend, _ = make_backend()

    capabilities = backend.capabilities
    # The served models come from the native engine registry rather than a
    # literal here, so a build that links a second engine reports it.
    assert capabilities.models == ("qwen3_5",)
    assert capabilities.model_formats == ("safetensors",)
    assert capabilities.supports_streaming is True
    assert capabilities.supports_batch is False


def test_cpp_capabilities_list_the_registered_architectures() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    assert backend.capabilities.models == ("deepseek_v4", "qwen3_5")
    backend.close()


def test_cpp_engine_kind_is_detected_from_the_checkpoint() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    # Detection ran against the checkpoint and picked the Qwen runtime, without
    # an engine_kind having been supplied.
    assert native.detected == ["checkpoint"]
    assert native.constructed is not None
    backend.close()


def test_cpp_unknown_architecture_is_reported_not_guessed() -> None:
    class UnknownArchModule(FakeNativeModule):
        def detect_architecture(self, checkpoint: str) -> str:
            return "llama"

    with pytest.raises(UnsupportedFeatureError, match="llama"):
        CppBackend(
            EngineArgs(model="checkpoint", backend="cpp"),
            native_module=UnknownArchModule(),
            tokenizer=FakeTokenizer(),
        )


def test_cpp_engine_kind_override_skips_detection() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(
            model="checkpoint",
            backend="cpp",
            backend_options={"engine_kind": "qwen"},
        ),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    assert native.detected == []
    backend.close()


def test_cpp_capabilities_follow_the_linked_device_backend() -> None:
    class AscendModule(FakeNativeModule):
        backend = "ascend"

    native = AscendModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    capabilities = backend.capabilities
    # The Ascend build rejects DSpark/DFlash2 drafters, so they must not be
    # advertised just because the CUDA build supports them.
    assert capabilities.supports_speculative_decoding == ("mtp",)
    assert capabilities.devices == ("ascend",)
    backend.close()


def test_cancel_only_accepts_active_request_ids() -> None:
    backend, _ = make_backend()

    assert backend.cancel("req-never-submitted") is False

    stream = backend.stream(
        GenerationRequest(
            prompt_tokens=[1],
            request_id="req-active",
            sampling_params=SamplingParams(max_tokens=2),
        )
    )
    assert next(stream).token_id == 10
    assert backend.cancel("req-active") is True
    assert backend.cancel("req-other") is False
    with pytest.raises(RequestCancelledError):
        next(stream)
    assert backend.cancel("req-active") is False
    assert backend.active_request_count() == 0


def test_stream_text_deltas_use_cumulative_tokenizer_decode() -> None:
    class MergingTokenizer:
        """Emulates a tokenizer whose pieces only render as a full sequence."""

        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, token_ids: list[int]) -> str:
            return "".join("ab"[index % 2] for index, _ in enumerate(token_ids))

    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=engine,
        tokenizer=MergingTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-delta",
        sampling_params=SamplingParams(max_tokens=3),
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["a", "b", "a"]
    backend.close()


def test_generate_stops_at_eos_and_reports_stop() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": 11}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-eos-generate",
        sampling_params=SamplingParams(max_tokens=4),
    )

    result = backend.generate([request])[0]

    # With a known EOS the adapter drives prefill/decode itself so the session is
    # never advanced past EOS, and EOS stays out of the visible output.
    assert engine.calls == [("prefill", [4, 5]), ("decode_step", 10)]
    assert result.token_ids == [10]
    assert result.text == "<10>"
    assert result.finish_reason == "stop"
    assert result.usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    backend.close()


def test_stream_stops_at_eos_without_another_decode_step() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": [12]}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-eos-stream",
        sampling_params=SamplingParams(max_tokens=8),
    )

    events = list(backend.stream(request))

    assert engine.calls == [
        ("prefill", [4, 5]),
        ("decode_step", 10),
        ("decode_step", 11),
    ]
    assert [event.token_id for event in events] == [10, 11, None]
    assert [event.text for event in events] == ["<10>", "<11>", ""]
    assert [event.finish_reason for event in events] == [None, None, "stop"]
    assert events[-1].usage is not None
    assert events[-1].usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    backend.close()


def test_generate_does_not_advance_the_session_past_eos() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": 11}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )

    backend.generate([
        GenerationRequest(
            prompt_tokens=[4],
            request_id="req-a",
            sampling_params=SamplingParams(max_tokens=6),
        )
    ])
    backend.generate([
        GenerationRequest(
            prompt_tokens=[5],
            request_id="req-b",
            sampling_params=SamplingParams(max_tokens=6),
        )
    ])

    # Each request stops at the EOS step, so no decode runs on text the caller
    # never received and the next request starts from a clean position.
    assert engine.calls == [
        ("prefill", [4]),
        ("decode_step", 10),
        ("prefill", [5]),
        ("decode_step", 10),
    ]
    backend.close()


def test_eos_comes_from_the_native_engine_when_available() -> None:
    class EosEngine(FakeEngine):
        def eos_id(self) -> int:
            return 11

    backend, _ = make_backend(EosEngine())

    assert backend.eos_token_ids == frozenset({11})
    assert backend.capabilities.details["eos_source"] == "native engine"
    backend.close()


def test_eos_falls_back_to_generation_config(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [7, 9]}', encoding="utf-8")
    (tmp_path / "config.json").write_text('{"eos_token_id": 1}', encoding="utf-8")

    backend = CppBackend(
        EngineArgs(model=str(tmp_path), backend="cpp"),
        engine=FakeEngine(),
        tokenizer=FakeTokenizer(),
    )

    # generation_config.json wins over config.json because it is what the
    # checkpoint declares for generation.
    assert backend.eos_token_ids == frozenset({7, 9})
    assert backend.capabilities.details["eos_source"] == "generation_config.json"
    backend.close()


def test_without_any_eos_generation_ends_on_the_token_budget() -> None:
    backend, _ = make_backend()

    assert backend.eos_token_ids == frozenset()
    assert backend.capabilities.details["eos_source"] == "none"
    result = backend.generate(
        [
            GenerationRequest(
                prompt_tokens=[1],
                request_id="req-no-eos",
                sampling_params=SamplingParams(max_tokens=2),
            )
        ]
    )[0]
    assert result.finish_reason == "length"
    backend.close()


def test_invalid_eos_override_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="eos_token_id"):
        CppBackend(
            EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": "11"}),
            engine=FakeEngine(),
            tokenizer=FakeTokenizer(),
        )


def test_cpp_backend_rejects_unexposed_sampling_controls() -> None:
    backend, _ = make_backend()
    request = GenerationRequest(
        prompt_tokens=[1],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.5),
    )

    with pytest.raises(UnsupportedFeatureError, match="greedy"):
        backend.generate([request])
