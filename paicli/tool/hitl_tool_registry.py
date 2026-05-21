"""Tool registry wrapper that intercepts dangerous calls for HITL approval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Callable, Iterable, Mapping

from paicli.hitl import ApprovalPolicy, ApprovalRequest, ApprovalResult, HitlHandler
from paicli.mcp.browser_session import BrowserMode, BrowserSession
from paicli.tool.tool_registry import CodeRetrieverFactory, ToolProvider, ToolRegistry


DEFAULT_SENSITIVE_PATTERNS = [
    "*://*.bank.*/*",
    "*://*.alipay.com/*",
    "*://alipay.com/*",
    "*://*.paypal.com/*",
    "*://paypal.com/*",
    "*://*.stripe.com/*",
    "*://stripe.com/*",
    "*://github.com/settings/*",
    "*://*.feishu.cn/admin/*",
    "*://*.larksuite.com/admin/*",
    "*://console.cloud.google.com/*",
    "*://*.console.cloud.google.com/*",
    "*://console.aws.amazon.com/*",
    "*://*.console.aws.amazon.com/*",
    "*://portal.azure.com/*",
    "*://*.portal.azure.com/*",
    "*://*.aliyun.com/*",
    "*://*.cloud.tencent.com/*",
]


@dataclass(frozen=True)
class BrowserCheckResult:
    allowed: bool
    message: str = ""


class SensitivePagePolicy:
    def __init__(
        self,
        patterns: Iterable[str] | None = None,
        custom_path: str | Path | None = None,
    ) -> None:
        base_patterns = list(DEFAULT_SENSITIVE_PATTERNS if patterns is None else patterns)
        if custom_path is None:
            custom_path = Path.home() / ".paicli" / "sensitive_patterns.txt"
        self.patterns = [*base_patterns, *_read_custom_patterns(Path(custom_path).expanduser())]

    def match(self, url: str) -> str | None:
        lowered = (url or "").lower()
        for pattern in self.patterns:
            if fnmatchcase(lowered, pattern.lower()):
                return pattern
        return None


class BrowserGuard:
    WRITE_TOOLS = {
        "click",
        "drag",
        "fill",
        "fill_form",
        "handle_dialog",
        "hover",
        "navigate_page",
        "new_page",
        "press_key",
        "resize_page",
        "upload_file",
        "evaluate_script",
    }
    READ_TOOLS = {
        "take_snapshot",
        "take_screenshot",
        "list_pages",
        "list_console_messages",
        "list_network_requests",
    }

    def __init__(
        self,
        session: BrowserSession,
        policy: SensitivePagePolicy | None = None,
    ) -> None:
        self.session = session
        self.policy = policy or SensitivePagePolicy()

    def check(self, tool_name: str, args: Mapping[str, str]) -> BrowserCheckResult:
        local_tool = _chrome_local_tool(tool_name)
        if not local_tool:
            return BrowserCheckResult(True)
        if (
            local_tool == "close_page"
            and self.session.mode == BrowserMode.SHARED
            and not self.session.is_agent_opened_tab(_page_id(args))
        ):
            return BrowserCheckResult(
                False,
                "shared 浏览器模式下拒绝关闭非 PaiCLI 创建的标签页，请手动关闭该 Chrome 标签页",
            )
        return BrowserCheckResult(True)

    def approval_request(self, tool_name: str, arguments_json: str) -> ApprovalRequest | None:
        local_tool = _chrome_local_tool(tool_name)
        if not local_tool or local_tool not in self.WRITE_TOOLS:
            return None
        current_url = _target_url(local_tool, arguments_json) or self.session.current_url
        matched = self.policy.match(current_url)
        if not matched:
            return None
        return ApprovalRequest.of(
            tool_name,
            arguments_json,
            suggestion="检测到敏感页面，本次操作需单独确认（不接受全部放行）",
            caller_context=f"当前 URL: {current_url}\n匹配规则: {matched}",
            allow_approve_all=False,
        )

    def apply_after_execution(
        self,
        tool_name: str,
        args: Mapping[str, str],
        result: str,
    ) -> str:
        local_tool = _chrome_local_tool(tool_name)
        if not local_tool:
            return ""
        if local_tool in {"new_page", "navigate_page"}:
            target = str(args.get("url") or "").strip()
            if target:
                self.session.current_url = target
        if local_tool == "new_page":
            self.session.record_opened_tab(_extract_page_id(result))
        if local_tool == "close_page":
            self.session.forget_opened_tab(_page_id(args))
        if self.session.mode == BrowserMode.ISOLATED and _looks_like_login_required(result):
            return "\n\n[Browser] 当前页面疑似需要登录态，正在切换到 shared(autoConnect) 模式，请重试同一页面。\n" + self.session.switch_to_shared()
        return ""


class HitlToolRegistry(ToolRegistry):
    def __init__(
        self,
        hitl_handler: HitlHandler,
        base_dir: str | Path | None = None,
        providers: Iterable[ToolProvider] | None = None,
        code_retriever_factory: CodeRetrieverFactory | None = None,
        browser_guard: BrowserGuard | None = None,
        on_browser_restart: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            base_dir=base_dir,
            providers=providers,
            code_retriever_factory=code_retriever_factory,
        )
        self.hitl_handler = hitl_handler
        self.browser_guard = browser_guard
        self.on_browser_restart = on_browser_restart

    def execute(self, name: str, args: Mapping[str, str]) -> str:
        if self.browser_guard is not None:
            check = self.browser_guard.check(name, args)
            if not check.allowed:
                return f"[BrowserGuard] 策略拒绝：{check.message}"

        arguments_json = json.dumps(dict(args), ensure_ascii=False)
        browser_request = (
            self.browser_guard.approval_request(name, arguments_json)
            if self.browser_guard is not None
            else None
        )
        if browser_request is not None:
            result = self._request_and_execute(name, args, arguments_json, browser_request)
            return self._apply_browser_after_execution(name, args, result)

        if not self.hitl_handler.is_enabled() or not ApprovalPolicy.requires_approval(name):
            result = super().execute(name, args)
            return self._apply_browser_after_execution(name, args, result)

        request = ApprovalRequest.of(name, arguments_json, None)
        result = self._request_and_execute(name, args, arguments_json, request)
        return self._apply_browser_after_execution(name, args, result)

    def _request_and_execute(
        self,
        name: str,
        args: Mapping[str, str],
        arguments_json: str,
        request: ApprovalRequest,
    ) -> str:
        result = self.hitl_handler.request_approval(request)

        if result.is_rejected():
            reason = result.reason or "用户拒绝了此操作"
            return f"[HITL] 操作已被拒绝：{reason}"
        if result.is_skipped():
            return "[HITL] 操作已被跳过"

        effective_arguments = result.effective_arguments(arguments_json)
        try:
            parsed_args = json.loads(effective_arguments)
        except json.JSONDecodeError as exc:
            return f"[HITL] 修改后的参数不是有效 JSON：{exc}"
        if not isinstance(parsed_args, dict):
            return "[HITL] 修改后的参数必须是 JSON 对象"

        normalized_args = {
            str(key): value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in parsed_args.items()
        }
        return super().execute(name, normalized_args)

    def _apply_browser_after_execution(
        self,
        name: str,
        args: Mapping[str, str],
        result: str,
    ) -> str:
        if self.browser_guard is None:
            return result
        extra = self.browser_guard.apply_after_execution(name, args, result)
        if extra and self.on_browser_restart is not None:
            self.on_browser_restart()
        return result + extra


def _read_custom_patterns(path: Path) -> list[str]:
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def _chrome_local_tool(tool_name: str) -> str:
    prefix = "mcp__chrome_devtools__"
    if not tool_name.startswith(prefix):
        return ""
    return tool_name[len(prefix):]


def _page_id(args: Mapping[str, str]) -> str:
    for key in ["pageId", "page_id", "pageIdx", "pageIndex", "index"]:
        value = args.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_page_id(result: str) -> str:
    patterns = [
        r'"pageId"\s*:\s*"?([^",}\s]+)',
        r'"pageIdx"\s*:\s*"?([^",}\s]+)',
        r"\bpage(?:\s+with)?\s+id\s+([A-Za-z0-9_-]+)",
        r"\bpageIdx\s*[:=]\s*([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, result or "", re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _target_url(local_tool: str, arguments_json: str) -> str:
    if local_tool not in {"navigate_page", "new_page"}:
        return ""
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("url") or "")


def _looks_like_login_required(result: str) -> bool:
    lowered = (result or "").lower()
    markers = [
        "sign in",
        "log in",
        "login",
        "登录",
        "请先登录",
        "unauthorized",
        "forbidden",
        "permission denied",
        "access denied",
        "需要权限",
        "没有权限",
        "sso",
    ]
    return any(marker in lowered for marker in markers)
