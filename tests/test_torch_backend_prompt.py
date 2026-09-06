"""Prompt rendering tests for the Torch adapter.

These inject a fake serving engine so no checkpoint, CUDA device, or real
runtime is required; only the payload the adapter builds is inspected.
"""

from __future__ import annotations

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.torch_backend import TorchBackend
from src.encoding.deepseek_v4 import encode_messages


class RecordingServingEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def submit(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"content": "ok", "prompt_tokens": 2, "completion_tokens": 1, "finish_reason": "stop"}

    def submit_stream(self, payload: dict):
        self.payloads.append(payload)
        yield {"type": "token", "token_ids": [11]}
        yield {"type": "done", "prompt_tokens": 2, "completion_tokens": [[11]], "finish_reason": "stop"}

    def close(self) -> None:
        pass


class RecordingTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return [11, 12]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class TemplateRecordingTokenizer(RecordingTokenizer):
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        super().__init__()
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
        return [91, 92]


def _backend(tokenizer: RecordingTokenizer) -> tuple[TorchBackend, RecordingServingEngine]:
    engine = RecordingServingEngine()
    backend = TorchBackend(
        EngineArgs(model="model", backend="torch"),
        runtime={"tokenizer": tokenizer, "model_id": "fake-model"},
        serving_engine=engine,
    )
    return backend, engine


def test_chat_metadata_is_rendered_with_the_deepseek_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-chat",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    backend.generate([request])

    assert tokenizer.encoded == [encode_messages(messages, thinking_mode="chat")]
    assert engine.payloads[-1]["messages"] == messages
    assert engine.payloads[-1]["thinking_mode"] == "chat"
    backend.close()


def test_thinking_mode_and_effort_reach_the_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-thinking",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "thinking", "reasoning_effort": "max"},
    )

    backend.generate([request])

    assert tokenizer.encoded == [encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")]
    assert engine.payloads[-1]["reasoning_effort"] == "max"
    backend.close()


def test_tokenizer_owned_template_takes_precedence():
    tokenizer = TemplateRecordingTokenizer()
    backend, _ = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-owned-template",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    assert backend._prompt_ids(request) == [91, 92]
    assert tokenizer.encoded == []
    assert tokenizer.template_calls[0][0] == messages
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    assert tokenizer.template_calls[0][1]["enable_thinking"] is False
    backend.close()


def test_raw_prompts_are_encoded_unchanged():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt="raw completion prompt",
        request_id="req-raw",
        sampling_params=SamplingParams(max_tokens=1),
    )

    backend.generate([request])

    assert tokenizer.encoded == ["raw completion prompt"]
    assert engine.payloads[-1]["messages"] == [{"role": "user", "content": "raw completion prompt"}]
    backend.close()


def test_prompt_tokens_bypass_the_tokenizer():
    tokenizer = TemplateRecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt_tokens=[7, 8],
        request_id="req-tokens",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": [{"role": "user", "content": "ignored"}]},
    )

    backend.generate([request])

    assert tokenizer.encoded == []
    assert tokenizer.template_calls == []
    assert engine.payloads[-1]["_prompt_ids"] == [7, 8]
    backend.close()


def test_streaming_uses_the_same_prompt_rendering():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    list(backend.stream(request))

    assert tokenizer.encoded[0] == encode_messages(messages, thinking_mode="chat")
    assert engine.payloads[-1]["_prompt_ids"] == [11, 12]
    backend.close()
