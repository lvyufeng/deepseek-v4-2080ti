import json

from src.encoding.deepseek_v4 import REASONING_EFFORT_PROMPTS, encode_messages
from src.server.openai import (
    _completion_response,
    _make_payload,
    _normalize_tool_calls,
    _prepare_messages,
    _StreamingDecoder,
    _thinking_config,
)


def test_prepare_messages_preserves_tool_role_and_tool_choice_required():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "summarize"},
        ],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
    }
    messages = _prepare_messages(body)
    assert messages[0]["role"] == "system"
    assert messages[0].get("tools") == body["tools"]
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "must call" in messages[-1]["content"]


def test_builtin_web_search_auto_does_not_force_search_intent():
    body = {
        "messages": [{"role": "user", "content": "搜索一下特朗普访华"}],
        "tools": [{"type": "function", "function": {"name": "builtin_web_search", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }
    messages = _prepare_messages(body)
    assert messages[0]["role"] == "system"
    assert messages[0].get("tools") == body["tools"]
    assert "must call" not in messages[-1]["content"]


def test_builtin_web_search_auto_does_not_force_ordinary_chinese_chat():
    body = {
        "messages": [{"role": "user", "content": "生成一些特殊表情，我看看现在还会不会出现�"}],
        "tools": [{"type": "function", "function": {"name": "builtin_web_search", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }
    messages = _prepare_messages(body)
    assert messages[0]["role"] == "system"
    assert messages[0].get("tools") == body["tools"]
    assert "must call" not in messages[-1]["content"]


def test_named_builtin_web_search_tool_choice_still_forces_tool():
    body = {
        "messages": [{"role": "user", "content": "搜索一下特朗普访华"}],
        "tools": [{"type": "function", "function": {"name": "builtin_web_search", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "builtin_web_search"}},
    }
    messages = _prepare_messages(body)
    assert messages[0]["role"] == "system"
    assert messages[0].get("tools") == body["tools"]
    assert "must call the tool named builtin_web_search" in messages[-1]["content"]


def test_tools_are_passed_through_unchanged_for_prompt():
    prepared_description = "Web search tool. Prepared queries: 把前面回复的特朗普访华信息里的�修复一下"
    parameters = {"type": "object", "properties": {"additionalContext": {"type": "string"}}}
    body = {
        "messages": [{"role": "user", "content": "继续"}],
        "tools": [
            {"type": "function", "function": {"name": "builtin_web_search", "description": prepared_description, "parameters": parameters}},
            {"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}},
        ],
        "tool_choice": "auto",
    }
    messages = _prepare_messages(body)
    assert messages[0].get("tools") == body["tools"]
    assert messages[0]["tools"][0]["function"]["description"] == prepared_description
    assert messages[0]["tools"] is body["tools"]


def test_auto_tools_are_attached_for_plain_chat():
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    }
    messages = _prepare_messages(body)
    assert messages[0]["role"] == "system"
    assert messages[0].get("tools") == body["tools"]



def test_tool_choice_none_does_not_attach_tools():
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "noop"}}],
        "tool_choice": "none",
    }
    messages = _prepare_messages(body)
    assert "tools" not in messages[0]
    assert "Do not call" in messages[0]["content"]


def test_thinking_config_accepts_openai_and_deepseek_effort_fields():
    assert _thinking_config({"messages": [{"role": "user", "content": "hello"}]}) == ("chat", None)
    assert _thinking_config({"reasoning": {"effort": "high"}}) == ("thinking", "high")
    assert _thinking_config({"reasoning": {"effort": "max"}}) == ("thinking", "max")
    assert _thinking_config({"reasoning_effort": "max"}) == ("thinking", "max")
    assert _thinking_config({"reasoning": {"effort": "low"}}) == ("thinking", "low")


def test_nested_reasoning_effort_overrides_top_level_effort():
    assert _thinking_config({"reasoning_effort": "max", "reasoning": {"effort": "medium"}}) == ("thinking", "medium")


def test_invalid_reasoning_effort_keeps_thinking_without_effort_prefix():
    assert _thinking_config({"reasoning_effort": "extreme"}) == ("thinking", None)


def test_make_payload_accepts_top_level_reasoning_effort_max():
    payload = _make_payload({"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "max"})
    assert payload["thinking_mode"] == "thinking"
    assert payload["reasoning_effort"] == "max"


def test_encode_messages_injects_prefix_only_for_reasoning_effort_max():
    messages = [{"role": "user", "content": "hello"}]

    def prompt(effort):
        return encode_messages(messages, thinking_mode="thinking", reasoning_effort=effort)

    assert REASONING_EFFORT_PROMPTS["max"] in prompt("max")
    assert REASONING_EFFORT_PROMPTS["high"] in prompt("high")
    assert REASONING_EFFORT_PROMPTS["max"] not in prompt("high")
    # DeepSeek-V4 only defines low/high/max; None and OpenAI's extra levels all
    # fall back to low, which prepends nothing at all.
    for effort in (None, "low", "minimal", "medium"):
        assert REASONING_EFFORT_PROMPTS["high"] not in prompt(effort)
        assert REASONING_EFFORT_PROMPTS["max"] not in prompt(effort)
    assert prompt(None) == prompt("low") == prompt("minimal") == prompt("medium")


def test_encode_messages_ignores_reasoning_effort_in_chat_mode():
    messages = [{"role": "user", "content": "hello"}]
    for effort in (None, "low", "high", "max"):
        chat_prompt = encode_messages(messages, thinking_mode="chat", reasoning_effort=effort)
        assert REASONING_EFFORT_PROMPTS["high"] not in chat_prompt
        assert REASONING_EFFORT_PROMPTS["max"] not in chat_prompt


def test_make_payload_accepts_common_openai_params():
    payload = _make_payload({
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 7,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "seed": 123,
        "stop": ["END"],
        "n": 2,
        "logprobs": True,
        "top_logprobs": 3,
        "parallel_tool_calls": False,
        "user": "u",
    })
    assert payload["max_tokens"] == 7
    assert payload["n"] == 2
    assert payload["stop"] == ["END"]
    assert payload["parallel_tool_calls"] is False
    assert payload["generation_options"]["top_p"] == 0.9
    assert payload["generation_options"]["top_k"] == 40
    assert payload["generation_options"]["seed"] == 123
    assert payload["generation_options"]["logprobs"] is True


def test_normalize_tool_calls_adds_ids_and_openai_shape():
    calls = _normalize_tool_calls([{"type": "function", "function": {"name": "weather", "arguments": {"city": "sh"}}}])
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "sh"}


def test_completion_response_includes_logprobs():
    runtime = {"model_id": "m"}
    payload = {"request_id": "chatcmpl_x", "top_logprobs": 1}
    result = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "content": "A",
        "reasoning_content": "",
        "tool_calls": [],
        "finish_reason": "stop",
        "token_texts": ["A"],
        "token_logprobs": [-0.2],
        "top_logprobs": [[{"token": "A", "logprob": -0.2}]],
        "timings": {},
    }
    response = _completion_response(runtime, payload, result)
    logprobs = response["choices"][0]["logprobs"]["content"]
    assert logprobs[0]["token"] == "A"
    assert logprobs[0]["top_logprobs"][0]["logprob"] == -0.2



def test_completion_response_multiple_choices_and_tool_finish():
    runtime = {"model_id": "m"}
    payload = {"request_id": "chatcmpl_x", "top_logprobs": None}
    result = [
        {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}],
            "finish_reason": "tool_calls",
            "timings": {"prefill_time": 1.0},
        },
        {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "content": "ok",
            "reasoning_content": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "timings": {"prefill_time": 1.0},
        },
    ]
    response = _completion_response(runtime, payload, result)
    assert len(response["choices"]) == 2
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert response["usage"]["completion_tokens"] == 6


class _CharTokenizer:
    eos_token_id = 0

    def __init__(self):
        self._table: dict[int, str] = {}

    def add(self, text: str) -> int:
        token_id = len(self._table) + 1
        self._table[token_id] = text
        return token_id

    def decode(self, token_ids):
        return "".join(self._table.get(int(t), "") for t in token_ids)


def test_streaming_decoder_hides_dsml_tool_call_prefix():
    from src.encoding.deepseek_v4 import dsml_token

    tokenizer = _CharTokenizer()
    chunks = ["hi", "\n", "\n", "<", dsml_token, "tool", "_calls", ">"]
    chunk_ids = [tokenizer.add(c) for c in chunks]
    decoder = _StreamingDecoder(tokenizer, "no_thinking")
    visible = ""
    for tid in chunk_ids:
        for delta in decoder.append([tid]):
            visible += delta.get("content", "")
    assert visible == "hi"


class _ResolvingTokenizer:
    eos_token_id = 0

    def decode(self, token_ids):
        ids = [int(t) for t in token_ids]
        if ids == [1]:
            return "�"
        if ids == [1, 2]:
            return "中"
        return ""


def test_streaming_decoder_holds_back_trailing_replacement_character():
    decoder = _StreamingDecoder(_ResolvingTokenizer(), "no_thinking")
    assert decoder.append([1]) == []
    assert decoder.append([2]) == [{"content": "中"}]
