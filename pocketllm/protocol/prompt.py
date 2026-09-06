"""Backend-neutral chat prompt encoding helpers.

Tokenization and chat-template application belong to the control plane rather
than to a device backend.  The helper keeps model-specific template behavior
behind the tokenizer supplied by the selected runtime, while preserving the
validated DeepSeek encoder for checkpoints that predate Hugging Face chat
metadata.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any


_TEMPLATE_REASONING_EFFORTS = {
    None: "xhigh",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "max": "xhigh",
}


def _template_reasoning_effort(value: Any) -> str:
    """Map public reasoning names to the names used by Qwen templates."""
    return _TEMPLATE_REASONING_EFFORTS.get(value, "xhigh")


def _template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages and make OpenAI tool arguments template-compatible.

    OpenAI carries function arguments as a JSON string.  Qwen's bundled
    template iterates over ``tool_call.function.arguments`` as an object, so
    only the private copy passed to the template converts valid JSON objects
    back to dictionaries.  The normalized request metadata is never mutated.
    """
    copied = copy.deepcopy([dict(message) for message in messages])
    for message in copied:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                decoded = json.loads(arguments)
            except (TypeError, ValueError):
                continue
            if isinstance(decoded, dict):
                function["arguments"] = decoded
    return copied


def _token_ids(value: Any, tokenizer: Any) -> list[int] | None:
    """Normalize list/tensor-like tokenizer output to public token ids."""
    if isinstance(value, str) or value is None:
        encoder = getattr(tokenizer, "encode", None)
        if not isinstance(value, str) or not callable(encoder):
            return None
        value = encoder(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
    if not isinstance(value, list):
        return None
    # A tokenizer may return a single [1, N] batch even when tokenize=True.
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not all(isinstance(item, (int, bool)) for item in value):
        return None
    return [int(item) for item in value]


def _encode_with_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str,
    reasoning_effort: Any,
    tools: Any,
) -> list[int] | None:
    # Hugging Face exposes ``apply_chat_template`` on many tokenizers even
    # when the checkpoint has no actual template.  Check the metadata first so
    # DeepSeek's legacy fallback is selected instead of raising from the HF
    # helper.
    if hasattr(tokenizer, "chat_template") and not getattr(tokenizer, "chat_template", None):
        return None
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None

    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": thinking_mode == "thinking",
        "reasoning_effort": _template_reasoning_effort(reasoning_effort),
    }
    if tools is not None:
        # A custom template is model code from the checkpoint; keep all public
        # request metadata private from accidental in-place mutations.
        kwargs["tools"] = copy.deepcopy(tools)
    template_messages = _template_messages(messages)
    try:
        encoded = apply_chat_template(template_messages, **kwargs)
    except TypeError:
        # Tokenizers with a valid template but an older or custom method may not
        # accept optional model-specific controls.  Preserve the core template
        # contract, dropping only keywords rejected by the callable's signature.
        try:
            signature = inspect.signature(apply_chat_template)
        except (TypeError, ValueError):
            return None
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            raise
        supported = {
            name for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        if "messages" not in supported:
            return None
        # The first positional argument is supplied explicitly; only retain
        # keyword controls that the callable can actually receive.
        supported.discard("messages")
        compatible_kwargs = {name: value for name, value in kwargs.items() if name in supported}
        try:
            encoded = apply_chat_template(template_messages, **compatible_kwargs)
        except TypeError:
            return None
    return _token_ids(encoded, tokenizer)


def _encode_with_deepseek(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str,
    reasoning_effort: Any,
) -> list[int] | None:
    """Use the legacy DeepSeek encoder only for no-template tokenizers."""
    try:
        from src.encoding.deepseek_v4 import encode_messages
    except Exception:
        return None
    text = encode_messages(
        [dict(message) for message in messages],
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        return None
    return _token_ids(encoder(text), tokenizer)


def encode_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str = "chat",
    reasoning_effort: Any = None,
    tools: Any = None,
    deepseek_fallback: bool = False,
) -> list[int] | None:
    """Encode normalized chat messages using the best available model format.

    A checkpoint-owned ``apply_chat_template`` always wins.  The legacy
    DeepSeek encoder is used only when ``deepseek_fallback`` is explicitly
    enabled by the DeepSeek adapter; generic callers otherwise receive
    ``None`` and may use their own fallback.  ``None`` means the tokenizer
    cannot encode this chat request.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("messages must be a non-empty sequence")
    normalized = [message for message in messages if isinstance(message, Mapping)]
    if not normalized:
        raise ValueError("messages must contain at least one object")
    encoded = _encode_with_template(
        tokenizer,
        normalized,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )
    if encoded is not None:
        return encoded
    if not deepseek_fallback:
        return None
    return _encode_with_deepseek(
        tokenizer,
        normalized,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
