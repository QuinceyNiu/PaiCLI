"""Lifecycle manager for configured MCP servers."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field

from paicli.mcp.client import McpClient, McpTool, create_mcp_client
from paicli.mcp.config import McpConfig, McpServerConfig


@dataclass
class McpServerRuntime:
    config: McpServerConfig
    status: str = "disabled"
    tools: list[McpTool] = field(default_factory=list)
    error: str = ""
    started_at: float | None = None
    client: McpClient | None = None

    @property
    def uptime_seconds(self) -> int:
        if self.started_at is None:
            return 0
        return int(time.monotonic() - self.started_at)

    @property
    def pid(self) -> int | None:
        transport = getattr(self.client, "transport", None)
        return getattr(transport, "pid", None)

    def logs(self) -> list[str]:
        transport = getattr(self.client, "transport", None)
        logs = getattr(transport, "logs", None)
        if callable(logs):
            return list(logs())
        return []


class McpServerManager:
    def __init__(self, config: McpConfig, client_factory=create_mcp_client) -> None:
        self.config = config
        self._client_factory = client_factory
        self._runtimes: dict[str, McpServerRuntime] = {
            server.name: McpServerRuntime(server) for server in config.servers
        }

    def start_all(self) -> None:
        targets = [
            runtime for runtime in self._runtimes.values()
            if runtime.config.enabled and runtime.status != "ready"
        ]
        if not targets:
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(targets), 8),
            thread_name_prefix="paicli-mcp-startup",
        ) as pool:
            list(pool.map(lambda runtime: self.start(runtime.config.name), targets))

    def start(self, name: str) -> None:
        runtime = self._runtimes.get(name)
        if runtime is None:
            return
        if not runtime.config.enabled:
            runtime.status = "disabled"
            return
        config_error = runtime.config.env.get("PAICLI_MCP_CONFIG_ERROR", "")
        if config_error:
            runtime.status = "error"
            runtime.error = config_error
            return
        try:
            client = self._client_factory(runtime.config)
            tools = client.list_tools()
            runtime.client = client
            runtime.tools = tools
            runtime.status = "ready"
            runtime.error = ""
            runtime.started_at = time.monotonic()
        except Exception as exc:
            runtime.status = "error"
            runtime.error = str(exc)
            runtime.tools = []

    def restart(self, name: str) -> str:
        runtime = self._runtimes.get(name)
        if runtime is None:
            return f"MCP server 不存在: {name}"
        self._close(runtime)
        runtime.status = "disabled" if not runtime.config.enabled else "starting"
        self.start(name)
        return f"MCP server {name} 已重启，当前状态: {runtime.status}"

    def enable(self, name: str) -> str:
        runtime = self._runtimes.get(name)
        if runtime is None:
            return f"MCP server 不存在: {name}"
        runtime.config = McpServerConfig(
            name=runtime.config.name,
            transport=runtime.config.transport,
            command=runtime.config.command,
            args=runtime.config.args,
            env=runtime.config.env,
            url=runtime.config.url,
            headers=runtime.config.headers,
            enabled=True,
        )
        self.start(name)
        return f"MCP server {name} 已启用，当前状态: {runtime.status}"

    def disable(self, name: str) -> str:
        runtime = self._runtimes.get(name)
        if runtime is None:
            return f"MCP server 不存在: {name}"
        self._close(runtime)
        runtime.status = "disabled"
        runtime.tools = []
        return f"MCP server {name} 已禁用"

    def logs(self, name: str) -> str:
        runtime = self._runtimes.get(name)
        if runtime is None:
            return f"MCP server 不存在: {name}"
        lines = runtime.logs()
        if not lines:
            return f"MCP server {name} 没有 stderr 日志"
        return "\n".join(lines[-200:])

    def runtimes(self) -> list[McpServerRuntime]:
        return list(self._runtimes.values())

    def ready_runtimes(self) -> list[McpServerRuntime]:
        return [runtime for runtime in self._runtimes.values() if runtime.status == "ready"]

    def close(self) -> None:
        for runtime in self._runtimes.values():
            self._close(runtime)

    def _close(self, runtime: McpServerRuntime) -> None:
        if runtime.client is not None:
            try:
                runtime.client.close()
            except Exception:
                pass
        runtime.client = None
