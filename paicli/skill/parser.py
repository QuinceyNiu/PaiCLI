"""Small frontmatter parser for SKILL.md files."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedSkillDocument:
    metadata: dict[str, Any]
    body: str


class SkillFrontmatterParser:
    """Parse the SKILL.md subset PaiCLI supports.

    Supported frontmatter values are single-line scalars, ``|`` block strings,
    and inline arrays. Unsupported fields are skipped with a warning.
    """

    def parse(self, document: str, skill_name: str = "") -> ParsedSkillDocument:
        lines = document.splitlines()
        if not lines or lines[0].strip() != "---":
            return ParsedSkillDocument({}, document)

        closing_index = None
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                closing_index = index
                break
        if closing_index is None:
            return ParsedSkillDocument({}, document)

        metadata = self._parse_frontmatter(lines[1:closing_index], skill_name or "unknown")
        body = "\n".join(lines[closing_index + 1 :])
        return ParsedSkillDocument(metadata, body)

    def _parse_frontmatter(self, lines: list[str], skill_name: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        index = 0
        while index < len(lines):
            raw = lines[index]
            line_number = index + 2
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if ":" not in raw:
                self._warn(skill_name, line_number, "缺少 ':'，已跳过")
                index += 1
                continue

            key, value = raw.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                self._warn(skill_name, line_number, "字段名为空，已跳过")
                index += 1
                continue

            if value == "|":
                block_lines: list[str] = []
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if continuation.strip() and not continuation.startswith((" ", "\t")):
                        break
                    block_lines.append(continuation[2:] if continuation.startswith("  ") else continuation.lstrip())
                    index += 1
                metadata[key] = "\n".join(block_lines).rstrip()
                continue

            parsed = self._parse_value(value)
            if parsed is _Unsupported:
                self._warn(skill_name, line_number, "包含不支持的语法，已跳过")
            else:
                metadata[key] = parsed
            index += 1
        return metadata

    def _parse_value(self, value: str) -> Any:
        if value.startswith("{") or (value.startswith("[") and not value.endswith("]")):
            return _Unsupported
        if value.startswith("["):
            if not value.endswith("]"):
                return _Unsupported
            inner = value[1:-1].strip()
            if not inner:
                return []
            items = [self._strip_quotes(item.strip()) for item in inner.split(",")]
            return [item for item in items if item]
        return self._strip_quotes(value)

    def _strip_quotes(self, value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    def _warn(self, skill_name: str, line_number: int, message: str) -> None:
        print(
            f"⚠️ Skill '{skill_name}' frontmatter 解析警告：第 {line_number} 行{message}",
            file=sys.stderr,
        )


class _UnsupportedValue:
    pass


_Unsupported = _UnsupportedValue()
