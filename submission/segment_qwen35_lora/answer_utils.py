"""Output controls shared by Qwen3.5 submission inference and local tests."""

from __future__ import annotations

import re
from typing import Any


_TIME_AT_START = re.compile(r"^\s*(\d{1,2}):([0-5]\d):([0-5]\d)(?!\d)")
_BOUNDARY_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "\nuser",
    "\nassistant",
    "<think>",
    "</think>",
)


def generation_eos_token_ids(tokenizer: Any) -> int | list[int] | None:
    """Return explicit chat-stop IDs, excluding unresolved unknown tokens."""
    token_ids: list[int] = []
    unknown_id = getattr(tokenizer, "unk_token_id", None)

    def add(value: Any) -> None:
        values = value if isinstance(value, (list, tuple)) else [value]
        for token_id in values:
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and token_id != unknown_id
                and token_id not in token_ids
            ):
                token_ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    try:
        add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    except Exception:
        pass

    if not token_ids:
        return None
    return token_ids[0] if len(token_ids) == 1 else token_ids


def clean_generated_answer(raw_answer: str) -> str:
    """Keep the concise answer and discard leaked chat/thinking continuations."""
    answer = raw_answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1].lstrip()

    marker_positions = [
        position
        for marker in _BOUNDARY_MARKERS
        if (position := answer.find(marker)) >= 0
    ]
    if marker_positions:
        answer = answer[: min(marker_positions)].strip()

    answer = re.sub(r"^(?:assistant|answer)\s*[:：]?\s*", "", answer, flags=re.I)
    first_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")

    time_match = _TIME_AT_START.match(first_line)
    if time_match:
        hour, minute, second = time_match.groups()
        return f"{int(hour):02d}:{minute}:{second}"
    return first_line
