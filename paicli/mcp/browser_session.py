"""Runtime browser session state for Chrome DevTools MCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from paicli.mcp.manager import McpServerManager


CHROME_DEVTOOLS_SERVER = "chrome-devtools"
CHROME_DEVTOOLS_PACKAGE_ARGS = ["-y", "chrome-devtools-mcp@latest"]
ISOLATED_ARGS = [*CHROME_DEVTOOLS_PACKAGE_ARGS, "--isolated=true"]
SHARED_AUTO_CONNECT_ARGS = [*CHROME_DEVTOOLS_PACKAGE_ARGS, "--autoConnect", "--channel=stable"]


class BrowserMode(str, Enum):
    ISOLATED = "isolated"
    SHARED = "shared"


class BrowserHitlHandler(Protocol):
    def clear_approved_all_for_server(self, server_name: str) -> None:
        """Clear approve-all entries for a server."""


@dataclass
class BrowserSession:
    manager: McpServerManager
    server_name: str = CHROME_DEVTOOLS_SERVER
    hitl_handler: BrowserHitlHandler | None = None
    mode: BrowserMode = BrowserMode.ISOLATED
    current_url: str = ""
    last_switch_error: str = ""
    agent_opened_tabs: set[str] = field(default_factory=set)

    def switch_to_shared(self, mode: str = "autoConnect") -> str:
        if mode != "autoConnect":
            return f"❌ 不支持的 shared 浏览器模式: {mode}"
        result = self.manager.restart_with_args(self.server_name, SHARED_AUTO_CONNECT_ARGS)
        runtime = self.manager.runtime(self.server_name)
        if runtime is not None and runtime.status == "ready":
            self.mode = BrowserMode.SHARED
            self.last_switch_error = ""
            if self.hitl_handler is not None:
                self.hitl_handler.clear_approved_all_for_server(self.server_name)
            return f"🌐 浏览器已切换到 shared(autoConnect) 模式。{result}"
        self.last_switch_error = result
        restore_result = self.manager.restart_with_args(self.server_name, ISOLATED_ARGS)
        self.mode = BrowserMode.ISOLATED
        return (
            "⚠️ shared 浏览器连接失败，继续使用 isolated 模式。"
            "请确认 Chrome 144+ 已在 chrome://inspect/#remote-debugging 开启远程调试，"
            "并在 Chrome 弹窗中允许 PaiCLI 连接。\n"
            f"{result}\n{restore_result}"
        )

    def disconnect(self) -> str:
        result = self.manager.restart_with_args(self.server_name, ISOLATED_ARGS)
        runtime = self.manager.runtime(self.server_name)
        if runtime is not None and runtime.status == "ready":
            self.mode = BrowserMode.ISOLATED
            self.current_url = ""
            self.agent_opened_tabs.clear()
            self.last_switch_error = ""
            return f"🌐 浏览器已切回 isolated 模式。{result}"
        self.last_switch_error = result
        return f"⚠️ isolated 浏览器重启失败。\n{result}"

    def status(self) -> str:
        runtime = self.manager.runtime(self.server_name)
        if runtime is None:
            return "🌐 Browser: chrome-devtools 未配置"
        lines = [
            "🌐 Browser 状态:",
            f"- mode: {self.mode.value}",
            f"- server: {runtime.status}",
            f"- args: {' '.join(runtime.config.args or [])}",
            f"- current_url: {self.current_url or '-'}",
            f"- PaiCLI tabs: {len(self.agent_opened_tabs)}",
        ]
        if self.last_switch_error:
            lines.append(f"- last_error: {self.last_switch_error}")
        return "\n".join(lines)

    def record_opened_tab(self, page_id: str | None) -> None:
        if page_id and page_id.strip():
            self.agent_opened_tabs.add(page_id.strip())

    def forget_opened_tab(self, page_id: str | None) -> None:
        if page_id:
            self.agent_opened_tabs.discard(page_id.strip())

    def is_agent_opened_tab(self, page_id: str | None) -> bool:
        return bool(page_id and page_id.strip() in self.agent_opened_tabs)
