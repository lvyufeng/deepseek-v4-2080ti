from __future__ import annotations

from pocketllm.protocol import encode_chat_prompt


class TemplateTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self, encoded=None) -> None:
        self.calls: list[tuple[list[dict], dict]] = []
        self.encoded = encoded if encoded is not None else [31, 32]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(([dict(message) for message in messages], dict(kwargs)))
        return self.encoded

    def encode(self, text: str) -> list[int]:
        return [len(text)]


class TensorLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def test_template_encoding_uses_generation_and_reasoning_controls():
    tokenizer = TemplateTokenizer(TensorLike([[31, 32]]))
    messages = [{"role": "user", "content": "hi"}]

    result = encode_chat_prompt(
        tokenizer,
        messages,
        thinking_mode="thinking",
        reasoning_effort="high",
    )

    assert result == [31, 32]
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            },
        )
    ]


def test_template_encoding_passes_tools_and_does_not_mutate_arguments():
    tokenizer = TemplateTokenizer()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "user", "content": "Weather?"},
    ]
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]

    assert encode_chat_prompt(tokenizer, messages, tools=tools) == [31, 32]
    called_messages, kwargs = tokenizer.calls[0]
    assert kwargs["tools"] == tools
    assert kwargs["tools"] is not tools
    assert called_messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'


def test_public_reasoning_aliases_map_to_template_vocabulary():
    for public, expected in (("minimal", "low"), ("low", "low"), ("medium", "medium"), ("max", "xhigh"), (None, "xhigh")):
        tokenizer = TemplateTokenizer()
        encode_chat_prompt(tokenizer, [{"role": "user", "content": "hi"}], reasoning_effort=public)
        assert tokenizer.calls[0][1]["reasoning_effort"] == expected
        assert tokenizer.calls[0][1]["enable_thinking"] is False


def test_template_with_legacy_signature_drops_unsupported_controls():
    class LegacyTokenizer:
        chat_template = "{{ messages }}"

        def __init__(self) -> None:
            self.calls: list[tuple[list[dict], dict]] = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.calls.append(([dict(message) for message in messages], {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }))
            return [41, 42]

    tokenizer = LegacyTokenizer()
    assert encode_chat_prompt(
        tokenizer,
        [{"role": "user", "content": "hi"}],
        thinking_mode="thinking",
        reasoning_effort="high",
        tools=[],
    ) == [41, 42]
    assert tokenizer.calls[0][1] == {"tokenize": True, "add_generation_prompt": True}


def test_template_type_error_from_callable_with_kwargs_is_not_hidden():
    class BrokenTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **kwargs):
            raise TypeError("template rendering failed")

    try:
        encode_chat_prompt(BrokenTokenizer(), [{"role": "user", "content": "hi"}])
    except TypeError as exc:
        assert str(exc) == "template rendering failed"
    else:
        raise AssertionError("template TypeError should be preserved")


def test_no_template_returns_none_without_deepseek_fallback():
    class GenericTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    assert encode_chat_prompt(GenericTokenizer(), [{"role": "user", "content": "hi"}]) is None


def test_no_template_uses_deepseek_fallback_when_requested():
    class DeepSeekTokenizer:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    messages = [{"role": "user", "content": "hi"}]
    result = encode_chat_prompt(
        DeepSeekTokenizer(),
        messages,
        deepseek_fallback=True,
    )

    from src.encoding.deepseek_v4 import encode_messages

    assert result == [len(encode_messages(messages, thinking_mode="chat", reasoning_effort=None))]
