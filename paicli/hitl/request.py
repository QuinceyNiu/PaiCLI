"""Approval request formatting for terminal display."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from paicli.hitl.policy import ApprovalPolicy


BOX_WIDTH = 58
VALUE_PREVIEW_CHARS = 120


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: str
    danger_level: str
    risk_description: str
    suggestion: str | None = None
    caller_context: str | None = None

    @classmethod
    def of(
        cls,
        tool_name: str,
        arguments: str,
        suggestion: str | None = None,
        caller_context: str | None = None,
    ) -> "ApprovalRequest":
        return cls(
            tool_name=tool_name,
            arguments=arguments,
            danger_level=ApprovalPolicy.get_danger_level(tool_name),
            risk_description=ApprovalPolicy.get_risk_description(tool_name),
            suggestion=suggestion,
            caller_context=caller_context,
        )

    def to_display_text(self) -> str:
        lines = [
            "┌" + "─" * BOX_WIDTH + "┐",
            _box_line("⚠️  需要审批"),
            "├" + "─" * BOX_WIDTH + "┤",
            _box_line(f"工具: {self.tool_name}"),
            _box_line(f"等级: {self.danger_level}"),
            _box_line(f"风险: {self.risk_description}"),
        ]
        if self.suggestion:
            lines.append(_box_line(f"建议: {self.suggestion}"))
        if self.caller_context:
            lines.append(_box_line(f"上下文: {self.caller_context}"))

        lines.extend(["├" + "─" * BOX_WIDTH + "┤", _box_line("参数:")])
        for line in self._format_arguments():
            lines.append(_box_line("  " + line))
        lines.append("└" + "─" * BOX_WIDTH + "┘")
        return "\n".join(lines)

    def _format_arguments(self) -> list[str]:
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError:
            return [_preview_value("raw", self.arguments)]
        if not isinstance(parsed, dict):
            return [_preview_value("value", parsed)]
        if not parsed:
            return ["{}"]
        return [_preview_value(str(key), value) for key, value in parsed.items()]


def display_width(text: str) -> int:
    width = 0
    previous_codepoint = 0
    for char in text:
        codepoint = ord(char)
        if codepoint == 0xFE0F:
            previous_codepoint = codepoint
            continue
        if unicodedata.combining(char):
            previous_codepoint = codepoint
            continue
        width += 2 if _is_wide_codepoint(codepoint, char) else 1
        previous_codepoint = codepoint
    return width


def _is_wide_codepoint(codepoint: int, char: str) -> bool:
    if unicodedata.east_asian_width(char) in {"F", "W"}:
        return True
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
    )


def _box_line(content: str) -> str:
    clipped = _fit_to_width(content, BOX_WIDTH - 4)
    padding = BOX_WIDTH - 2 - display_width(clipped)
    return "│  " + clipped + " " * padding + "│"


def _fit_to_width(content: str, max_width: int) -> str:
    result: list[str] = []
    width = 0
    for char in content:
        char_width = display_width(char)
        if width + char_width > max_width:
            break
        result.append(char)
        width += char_width
    return "".join(result)


def _preview_value(key: str, value: Any) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, ensure_ascii=False)

    suffix = ""
    if len(rendered) > VALUE_PREVIEW_CHARS:
        suffix = f" ({len(rendered)} 字符)"
        rendered = rendered[:24] + "..."
    return f'{key}: "{rendered}"{suffix}'
