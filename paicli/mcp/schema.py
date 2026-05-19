"""Sanitize MCP input schemas for LLM function calling."""

from __future__ import annotations

from typing import Any, Mapping


DROP_KEYS = {"$schema", "$id", "$ref"}
COMBINER_KEYS = {"anyOf", "oneOf"}
MAX_DESCRIPTION_CHARS = 1000


class McpSchemaSanitizer:
    @classmethod
    def sanitize(cls, schema: Mapping[str, Any] | None) -> dict[str, Any]:
        cleaned = cls._clean(dict(schema or {}))
        if cleaned.get("type") != "object":
            if "properties" in cleaned and "type" not in cleaned:
                cleaned["type"] = "object"
            else:
                cleaned = {
                    "type": "object",
                    "properties": {"value": cleaned},
                    "required": ["value"],
                }
        cleaned.setdefault("properties", {})
        cleaned.setdefault("required", [])
        return cleaned

    @classmethod
    def _clean(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._clean(item) for item in value]
        if not isinstance(value, dict):
            return value

        output: dict[str, Any] = {}
        combiner_descriptions: list[str] = []
        for key, item in value.items():
            if key in DROP_KEYS:
                continue
            if key in COMBINER_KEYS:
                combiner_descriptions.append(cls._describe_combiner(key, item))
                continue
            if key == "description" and isinstance(item, str):
                output[key] = item[:MAX_DESCRIPTION_CHARS]
                continue
            output[key] = cls._clean(item)

        if combiner_descriptions:
            existing = str(output.get("description") or "")
            combined = " ".join([existing, *combiner_descriptions]).strip()
            output["description"] = combined[:MAX_DESCRIPTION_CHARS]
        return output

    @classmethod
    def _describe_combiner(cls, key: str, value: Any) -> str:
        options = value if isinstance(value, list) else []
        summaries = []
        for option in options:
            if not isinstance(option, dict):
                continue
            title = option.get("title")
            type_name = option.get("type")
            description = option.get("description")
            parts = [str(part) for part in [title, type_name, description] if part]
            if parts:
                summaries.append(" / ".join(parts))
        if not summaries:
            return f"{key}: multiple accepted shapes."
        return f"{key}: " + "; ".join(summaries)
