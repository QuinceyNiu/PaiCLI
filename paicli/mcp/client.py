"""Small JSON-RPC clients for MCP stdio and Streamable HTTP transports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import httpx

from paicli.mcp.config import McpServerConfig
from paicli.mcp.transport import McpTransport, StdioTransport, StreamableHttpTransport


JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"
INITIALIZE_TIMEOUT_SECONDS = 30.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "McpTool":
        schema = data.get("inputSchema") or data.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        return cls(
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            input_schema=schema,
        )


class McpClient(Protocol):
    def list_tools(self) -> list[McpTool]:
        """Return tools exposed by the MCP server."""

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Call an MCP tool and return text suitable for the agent observation."""


class JsonRpcMcpClient:
    def __init__(self, transport: McpTransport) -> None:
        self.transport = transport
        self._next_id = 1
        self._initialized = False

    def list_tools(self) -> list[McpTool]:
        self._ensure_initialized()
        result = self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, Mapping) else []
        return [McpTool.from_dict(tool) for tool in tools or []]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        self._ensure_initialized()
        result = self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments)},
        )
        return format_mcp_tool_result(result)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "paicli", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        self._initialized = True

    def _request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        response = self.transport.send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "method": method,
                "params": dict(params or {}),
            },
            timeout_seconds=INITIALIZE_TIMEOUT_SECONDS
            if method == "initialize"
            else DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        if response is None:
            raise RuntimeError(f"MCP 请求没有响应: {method}")
        if "error" in response:
            error = response["error"]
            if isinstance(error, Mapping):
                raise RuntimeError(error.get("message") or error)
            raise RuntimeError(error)
        result = response.get("result") or {}
        if not isinstance(result, dict):
            return {"value": result}
        return result

    def _notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self.transport.send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "method": method,
                "params": dict(params or {}),
            },
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self.transport.close()


class McpHttpClient(JsonRpcMcpClient):
    def __init__(
        self,
        config: McpServerConfig,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not config.url:
            raise ValueError("HTTP MCP server 缺少 url")
        self.config = config
        super().__init__(
            StreamableHttpTransport(
                config,
                protocol_version=MCP_PROTOCOL_VERSION,
                http_client=http_client,
            )
        )


class McpStdioClient(JsonRpcMcpClient):
    def __init__(self, config: McpServerConfig, timeout_seconds: float = 60.0) -> None:
        if not config.command:
            raise ValueError("stdio MCP server 缺少 command")
        self.config = config
        self.timeout_seconds = timeout_seconds
        super().__init__(StdioTransport(config))


def create_mcp_client(config: McpServerConfig) -> McpClient:
    if config.transport == "http":
        return McpHttpClient(config)
    return McpStdioClient(config)


def format_mcp_tool_result(result: Mapping[str, Any]) -> str:
    if "content" in result and isinstance(result["content"], list):
        parts = [_format_content_item(item) for item in result["content"]]
        return "\n".join(part for part in parts if part)
    if "structuredContent" in result:
        return json.dumps(result["structuredContent"], ensure_ascii=False, indent=2)
    if "value" in result:
        return json.dumps(result["value"], ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _format_content_item(item: Any) -> str:
    if isinstance(item, Mapping):
        if item.get("type") == "text":
            return str(item.get("text") or "")
        if item.get("type") and item.get("type") not in {"text", "json"}:
            return f"[非文本 MCP 内容: {item.get('type')}]"
        if "text" in item:
            return str(item["text"])
        if "json" in item:
            return json.dumps(item["json"], ensure_ascii=False, indent=2)
        return json.dumps(dict(item), ensure_ascii=False, indent=2)
    return str(item)
