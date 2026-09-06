#!/usr/bin/env python3
"""Check the Torch adapter renders the same prompt ids as the legacy server.

Phase 1.1 moved OpenAI request normalization into ``pocketllm.protocol`` and made
``TorchBackend`` apply the DeepSeek chat template itself.  The unified server
previously flattened messages to "role: content", so the prompt it fed the model
differed from ``src.server.openai``.  This drives real request bodies through
both paths using the real checkpoint's tokenizer and compares the resulting token
ids, which is the part of the Torch adapter that can be validated without the
full 156G model on device.

Usage:
  python scripts/verify_torch_prompt_parity.py --ckpt /mnt/data3/DeepSeek-V4-Flash-0731
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pocketllm.api import GenerationRequest, SamplingParams  # noqa: E402
from pocketllm.protocol import ChatRequest  # noqa: E402
from src.encoding.deepseek_v4 import encode_messages  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

BODIES: list[tuple[str, dict]] = [
    (
        "plain chat",
        {"messages": [{"role": "user", "content": "用一句话介绍你自己。"}]},
    ),
    (
        "system + multi turn",
        {
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ]
        },
    ),
    (
        "content blocks",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Name three prime numbers."},
                        {"type": "text", "text": " Answer with digits only."},
                    ],
                }
            ]
        },
    ),
    (
        "thinking mode",
        {
            "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
            "thinking": True,
            "reasoning_effort": "high",
        },
    ),
    (
        "tools + tool_choice",
        {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    ),
    (
        "tool result turn",
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "18C, clear"},
            ],
            "tools": [WEATHER_TOOL],
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    # Build a TorchBackend without loading the model: only prompt rendering and
    # the tokenizer are exercised here, so the runtime loader is stubbed out.
    from pocketllm.api import EngineArgs  # noqa: PLC0415
    from pocketllm.backends.torch_backend import TorchBackend  # noqa: PLC0415

    backend = TorchBackend(
        EngineArgs(model=args.ckpt, backend="torch"),
        runtime={"tokenizer": tokenizer, "model": None, "executor": None},
    )

    cases: list[dict] = []
    all_ok = True
    for label, body in BODIES:
        chat = ChatRequest.from_body(body)
        # Legacy path: src.server.openai encodes the normalized messages with the
        # DeepSeek template and tokenizes the result.
        legacy_text = encode_messages(
            chat.messages,
            thinking_mode=chat.thinking_mode,
            reasoning_effort=chat.reasoning_effort,
        )
        legacy_ids = [int(token) for token in tokenizer.encode(legacy_text)]

        # Unified path: the adapter renders from request metadata.
        request = GenerationRequest(
            prompt=chat.messages[-1].get("content", "") if chat.messages else "",
            request_id=f"prompt-parity-{len(cases)}",
            sampling_params=SamplingParams(max_tokens=8),
            metadata=chat.metadata(),
        )
        adapter_ids = backend._prompt_ids(request)

        ids_match = adapter_ids == legacy_ids
        # The shared boundary returns ids, while each backend may retain its
        # own textual rendering for runtime payloads.  Token ids are the
        # authoritative parity check because those are what the model sees.
        flattened = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in chat.messages
        )
        fallback_ids = [int(token) for token in tokenizer.encode(flattened)]
        differs_from_flattened = adapter_ids != fallback_ids

        passed = ids_match and differs_from_flattened
        all_ok = all_ok and passed
        checks = {
            "ids_match_legacy": ids_match,
            "differs_from_old_flattening": differs_from_flattened,
        }
        cases.append(
            {
                "label": label,
                "prompt_tokens": len(adapter_ids),
                "legacy_tokens": len(legacy_ids),
                "checks": checks,
                "pass": passed,
                "fallback_tokens": len(fallback_ids),
            }
        )
        print(f"[{label}] adapter={len(adapter_ids)} legacy={len(legacy_ids)} tokens", flush=True)
        print(f"  {'PASS' if passed else 'FAIL'} {checks}", flush=True)
        if not passed:
            print(f"  adapter_ids={adapter_ids[:32]}", flush=True)
            print(f"  legacy_ids ={legacy_ids[:32]}", flush=True)

    payload = {"ckpt": args.ckpt, "cases": cases, "pass": all_ok}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)

    print("", flush=True)
    if all_ok:
        print(f"ALL PASS ({len(cases)} cases)", flush=True)
        return 0
    failures = sum(1 for case in cases if not case["pass"])
    print(f"{failures} CASE(S) FAILED ({len(cases)} cases)", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
