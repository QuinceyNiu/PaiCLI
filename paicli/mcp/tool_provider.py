"""Expose MCP tools through PaiCli's tool registry."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable, Mapping

from paicli.mcp.audit import write_mcp_audit
from paicli.mcp.client import McpClient, create_mcp_client
from paicli.mcp.config import McpConfig, McpServerConfig
from paicli.mcp.manager import McpServerManager
from paicli.mcp.schema import McpSchemaSanitizer
from paicli.tool.tool_registry import RegisteredTool


McpClientFactory = Callable[[McpServerConfig], McpClient]


class McpToolProvider:
    def __init__(
        self,
        config: McpConfig | None = None,
        manager: McpServerManager | None = None,
        client_factory: McpClientFactory = create_mcp_client,
    ) -> None:
        self.config = config or McpConfig([])
        self.manager = manager
        self._client_factory = client_factory
        self._clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()

    def get_tools(self) -> list[RegisteredTool]:
        tools: list[RegisteredTool] = []
        if self.manager is not None:
            for runtime in self.manager.ready_runtimes():
                client = runtime.client
                if client is None:
                    continue
                for mcp_tool in runtime.tools:
                    tools.append(self._registered_tool(runtime.config, client, mcp_tool))
            return tools

        for server in self.config.servers:
            try:
                client = self._client_for(server)
                for mcp_tool in client.list_tools():
                    tools.append(self._registered_tool(server, client, mcp_tool))
            except Exception as exc:
                tools.append(
                    RegisteredTool(
                        name=f"mcp__{normalize_tool_name(server.name)}__status",
                        description=f"MCP server {server.name} 连接状态检查",
                        parameters={"type": "object", "properties": {}, "required": []},
                        executor=lambda _args, name=server.name, error=exc: (
                            f"MCP server {name} 暂不可用: {error}"
                        ),
                    )
                )
        return tools

    def _registered_tool(self, server, client, mcp_tool) -> RegisteredTool:
        safe_tool_name = normalize_tool_name(mcp_tool.name)
        registered_name = f"mcp__{normalize_tool_name(server.name)}__{safe_tool_name}"
        return RegisteredTool(
            name=registered_name,
            description=(
                f"MCP 工具 {server.name}.{mcp_tool.name}: "
                f"{mcp_tool.description}"
            ),
            parameters=McpSchemaSanitizer.sanitize(mcp_tool.input_schema),
            executor=_make_executor(client, mcp_tool.name, registered_name),
        )

    def _client_for(self, server: McpServerConfig) -> McpClient:
        with self._lock:
            if server.name not in self._clients:
                self._clients[server.name] = self._client_factory(server)
            return self._clients[server.name]


def normalize_tool_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    return normalized or "tool"


def _make_executor(client: McpClient, tool_name: str, registered_name: str):
    def execute(args: Mapping[str, str]) -> str:
        parsed_args: dict[str, Any] = {}
        for key, value in args.items():
            parsed_args[key] = _parse_argument(value)
        started_at = time.perf_counter()
        result = client.call_tool(tool_name, parsed_args)
        elapsed_millis = int((time.perf_counter() - started_at) * 1000)
        write_mcp_audit(registered_name, parsed_args, result, elapsed_millis)
        return result

    return execute


def _parse_argument(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
