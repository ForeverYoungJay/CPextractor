from __future__ import annotations

import json
import math
from typing import Any


def sanitize_text_for_openai(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return "".join(ch if (ch >= " " or ch in "\n\r\t") else " " for ch in text)


def sanitize_for_openai(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text_for_openai(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {
            sanitize_text_for_openai(str(k)): sanitize_for_openai(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_openai(v) for v in value]
    return value


def validate_openai_json_payload(payload: Any) -> Any:
    sanitized = sanitize_for_openai(payload)
    json.dumps(sanitized, ensure_ascii=False, allow_nan=False)
    return sanitized
