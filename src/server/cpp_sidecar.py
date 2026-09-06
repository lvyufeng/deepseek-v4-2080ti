"""Long-lived Python helper for the cpp_engine OpenAI server.

Reads JSON-line requests on stdin, writes JSON-line responses on stdout.
Reuses src.encoding.deepseek_v4 to render DeepSeek-V4 chat templates (with DSML
tool-call syntax, thinking_mode, reasoning_effort) and to parse generated
text back into {content, reasoning, tool_calls}.

Protocol (one JSON object per stdin/stdout line):

  request {"op": "encode", "messages": [...], "thinking_mode": "chat"|"thinking",
           "reasoning_effort": "low"|"medium"|"high"|null,
           "add_generation_prompt": true,
           "context": [...] (optional), "drop_thinking": true}
  reply   {"ok": true, "prompt_text": "...", "token_ids": [...]}

  request {"op": "parse", "text": "...", "thinking_mode": "chat"|"thinking"}
  reply   {"ok": true, "content": "...", "reasoning": "...", "tool_calls": [...]}

  request {"op": "ping"}
  reply   {"ok": true, "pong": true}

Errors: {"ok": false, "err": "..."}.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

from transformers import AutoTokenizer

# stdin/stdout must be UTF-8 regardless of locale — the C++ parent emits raw
# multi-byte JSON for non-ASCII content (Chinese prompts, tool-call XML, etc.).
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8")

# Make sure src/ is importable when invoked from arbitrary CWDs.
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.encoding.deepseek_v4 import encode_messages, eos_token, parse_message_from_completion_text  # noqa: E402


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    _emit({"ok": False, "err": msg})


def _handle_encode(tokenizer, req: dict[str, Any]) -> None:
    messages = list(req.get("messages") or [])
    thinking_mode = req.get("thinking_mode", "chat")
    reasoning_effort = req.get("reasoning_effort")
    context = req.get("context")
    drop_thinking = bool(req.get("drop_thinking", True))
    add_bos = bool(req.get("add_generation_prompt", True))
    tools = req.get("tools")
    if isinstance(tools, list) and tools:
        # Splice tools onto an existing system/developer message, or prepend one.
        attach_idx = None
        for idx, msg in enumerate(messages):
            if msg.get("role") in {"system", "developer"}:
                attach_idx = idx
                break
        if attach_idx is None:
            messages.insert(0, {"role": "system", "content": ""})
            attach_idx = 0
        messages[attach_idx] = {**messages[attach_idx], "tools": tools}
    prompt_text = encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_bos,
        reasoning_effort=reasoning_effort,
    )
    token_ids = tokenizer.encode(prompt_text)
    _emit({"ok": True, "prompt_text": prompt_text, "token_ids": list(token_ids)})


def _handle_parse(req: dict[str, Any]) -> None:
    text = req.get("text", "")
    thinking_mode = req.get("thinking_mode", "chat")
    # parse_message_from_completion_text requires the EOS token to be present.
    # The C++ engine emits raw decoded text without re-inserting the EOS string
    # (it stops on the EOS token id), so we append it here if missing.
    if not text.endswith(eos_token):
        text = text + eos_token
    parsed = parse_message_from_completion_text(text, thinking_mode)
    _emit({"ok": True, **parsed})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (used to load the HF tokenizer)")
    parser.add_argument("--tokenizer-path", default=None, help="Optional override for tokenizer directory")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _emit({"ok": True, "ready": True, "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 1})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as ex:
            _err(f"json decode failed: {ex}")
            continue
        op = req.get("op")
        try:
            if op == "encode":
                _handle_encode(tokenizer, req)
            elif op == "parse":
                _handle_parse(req)
            elif op == "ping":
                _emit({"ok": True, "pong": True})
            elif op == "shutdown":
                _emit({"ok": True, "bye": True})
                return 0
            else:
                _err(f"unknown op: {op!r}")
        except Exception as ex:  # noqa: BLE001
            _err(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
