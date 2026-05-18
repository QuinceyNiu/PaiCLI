"""Basic memory entry model and token estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MemoryType(str, Enum):
    CONVERSATION = "CONVERSATION"
    FACT = "FACT"
    SUMMARY = "SUMMARY"
    TOOL_RESULT = "TOOL_RESULT"


def estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0

    chinese_chars = sum(1 for char in text if "\u4e00" < char < "\u9fff")
    other_chars = len(text) - chinese_chars
    return math.ceil(chinese_chars / 1.5 + other_chars / 4.0)


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    content: str
    type: MemoryType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = field(default_factory=dict)
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_count", estimate_tokens(self.content))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        timestamp_text = str(data["timestamp"])
        timestamp = datetime.fromisoformat(timestamp_text)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            type=MemoryType(str(data["type"])),
            timestamp=timestamp,
            metadata={
                str(key): str(value)
                for key, value in (data.get("metadata") or {}).items()
            },
        )
