"""HITL handler interfaces and terminal implementation."""

from __future__ import annotations

import sys
from threading import Lock, RLock
from typing import Protocol, TextIO

from paicli.hitl.request import ApprovalRequest
from paicli.hitl.result import ApprovalResult


class HitlHandler(Protocol):
    def is_enabled(self) -> bool:
        """Return whether HITL approval is currently enabled."""

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable HITL approval."""

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        """Ask a human to decide whether a tool call may proceed."""


class TerminalHitlHandler:
    def __init__(
        self,
        enabled: bool = False,
        input_stream: TextIO | None = None,
        output: TextIO | None = None,
    ) -> None:
        self._enabled = enabled
        self._input = input_stream or sys.stdin
        self._output = output or sys.stdout
        self._approved_all_tools: set[str] = set()
        self._approval_lock = RLock()
        self._state_lock = Lock()

    def is_enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self._enabled = enabled

    def clear_approved_all(self) -> None:
        with self._state_lock:
            self._approved_all_tools.clear()

    def clear_approved_all_for_server(self, server_name: str) -> None:
        normalized = server_name.replace("-", "_")
        prefix = f"mcp__{normalized}__"
        with self._state_lock:
            self._approved_all_tools = {
                tool for tool in self._approved_all_tools if not tool.startswith(prefix)
            }

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        with self._approval_lock:
            with self._state_lock:
                approved_all = (
                    request.tool_name in self._approved_all_tools
                    or _mcp_server_approval_key(request.tool_name) in self._approved_all_tools
                )
            if approved_all:
                self._println(f"  [HITL] {request.tool_name} 已在本次会话中全部放行，自动通过")
                return ApprovalResult.approve_all()

            self._println()
            self._println("────────── ⚠️  HITL 审批请求 ──────────")
            self._println(request.to_display_text())
            return self._prompt_until_decision(request)

    def _prompt_until_decision(self, request: ApprovalRequest) -> ApprovalResult:
        for _ in range(5):
            if request.allow_approve_all:
                self._println("请选择操作：[y/Enter] 批准  [a] 全部放行  [n] 拒绝  [s] 跳过  [m] 修改参数")
            else:
                self._println("请选择操作：[y/Enter] 批准  [n] 拒绝  [s] 跳过  [m] 修改参数")
            self._output.write("> ")
            self._output.flush()

            line = self._input.readline()
            if line == "":
                return ApprovalResult.reject("输入流已关闭")
            normalized = line.strip().lower()
            if normalized in {"", "y"}:
                return ApprovalResult.approve()
            if normalized == "a" and request.allow_approve_all:
                with self._state_lock:
                    self._approved_all_tools.add(
                        _mcp_server_approval_key(request.tool_name) or request.tool_name
                    )
                return ApprovalResult.approve_all()
            if normalized == "n":
                reason = self._read_extra_line("请输入拒绝原因（可留空）：")
                return ApprovalResult.reject(reason or "用户拒绝了此操作")
            if normalized == "s":
                return ApprovalResult.skip()
            if normalized == "m":
                modified = self._read_extra_line("请输入修改后的 JSON 参数：")
                return ApprovalResult.modify(modified)
            choices = "y/a/n/s/m" if request.allow_approve_all else "y/n/s/m"
            self._println(f"  ❓ 无法识别的选项，请输入 {choices} 之一")
        return ApprovalResult.reject("连续多次无效输入")

    def _read_extra_line(self, prompt: str) -> str:
        self._output.write(prompt)
        self._output.flush()
        return self._input.readline().strip()

    def _println(self, text: str = "") -> None:
        print(text, file=self._output)
        self._output.flush()


def _mcp_server_approval_key(tool_name: str) -> str:
    parts = (tool_name or "").split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return f"mcp__{parts[1]}__*"
    return ""
