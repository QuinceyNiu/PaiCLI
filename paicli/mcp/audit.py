"""Audit logging for third-party MCP tool calls."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Mapping


AUDIT_LOG_PATH = Path.home() / ".paicli" / "audit.log"
SENSITIVE_KEY_PATTERN = re.compile(r"(authorization|password|token|api[_-]?key|secret)", re.I)
BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.I)


def write_mcp_audit(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: str,
    elapsed_millis: int,
    path: Path = AUDIT_LOG_PATH,
) -> None:
    entry = {
        "timestamp": int(time.time()),
        "tool": tool_name,
        "arguments": redact(arguments),
        "result": redact_text(result),
        "elapsed_millis": elapsed_millis,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY_PATTERN.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str) -> str:
    return BEARER_PATTERN.sub("Bearer [REDACTED]", value)
