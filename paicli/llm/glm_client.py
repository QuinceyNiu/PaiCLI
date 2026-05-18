"""GLM-5.1 chat client with OpenAI-compatible tool calling support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.1"


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionCall":
        return cls(
            name=str(data.get("name", "")),
            arguments=str(data.get("arguments", "")),
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=str(data.get("id", "")),
            type=str(data.get("type", "function")),
            function=FunctionCall.from_dict(data.get("function") or {}),
        )


@dataclass(frozen=True)
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str],
        tool_calls: Optional[list[ToolCall]] = None,
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=str(data.get("role", "assistant")),
            content=data.get("content"),
            tool_calls=[
                ToolCall.from_dict(tool_call)
                for tool_call in data.get("tool_calls") or []
            ],
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "Usage":
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
        )


@dataclass(frozen=True)
class ChatResponse:
    message: Message
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)


class GLMClient:
    def __init__(
        self,
        api_key: str,
        api_url: str = API_URL,
        model: str = MODEL,
        connect_timeout: float = 60.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = (connect_timeout, read_timeout)

    def build_chat_payload(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
        }
        if tools:
            payload["tools"] = [self._tool_to_dict(tool) for tool in tools]
        return payload

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
    ) -> ChatResponse:
        import httpx

        payload = self.build_chat_payload(messages, tools)
        timeout = httpx.Timeout(
            timeout=None,
            connect=self.timeout[0],
            read=self.timeout[1],
            write=self.timeout[1],
            pool=self.timeout[0],
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return self.parse_chat_response(response.json())

    def parse_chat_response(self, data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("GLM response did not include choices")

        choice = choices[0]
        return ChatResponse(
            message=Message.from_dict(choice.get("message") or {}),
            finish_reason=choice.get("finish_reason"),
            usage=Usage.from_dict(data.get("usage")),
            raw=data,
        )

    def _message_to_dict(self, message: Message) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": message.role,
            "content": self._sanitize_text(message.content),
        }
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": self._sanitize_text(tool_call.id),
                    "type": self._sanitize_text(tool_call.type),
                    "function": {
                        "name": self._sanitize_text(tool_call.function.name),
                        "arguments": self._sanitize_text(tool_call.function.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ]
        if message.tool_call_id:
            data["tool_call_id"] = self._sanitize_text(message.tool_call_id)
        return data

    def _tool_to_dict(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._sanitize_text(tool.name),
                "description": self._sanitize_text(tool.description),
                "parameters": self._sanitize_json_value(tool.parameters),
            },
        }

    def _sanitize_json_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_json_value(item) for item in value]
        if isinstance(value, dict):
            return {
                self._sanitize_text(str(key)): self._sanitize_json_value(item)
                for key, item in value.items()
            }
        return value

    def _sanitize_text(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return "".join(
            "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
            for character in value
        )
