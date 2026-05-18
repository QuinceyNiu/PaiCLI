"""Tool registry wrapper that intercepts dangerous calls for HITL approval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from paicli.hitl import ApprovalPolicy, ApprovalRequest, ApprovalResult, HitlHandler
from paicli.tool.tool_registry import CodeRetrieverFactory, ToolProvider, ToolRegistry


class HitlToolRegistry(ToolRegistry):
    def __init__(
        self,
        hitl_handler: HitlHandler,
        base_dir: str | Path | None = None,
        providers: Iterable[ToolProvider] | None = None,
        code_retriever_factory: CodeRetrieverFactory | None = None,
    ) -> None:
        super().__init__(
            base_dir=base_dir,
            providers=providers,
            code_retriever_factory=code_retriever_factory,
        )
        self.hitl_handler = hitl_handler

    def execute(self, name: str, args: Mapping[str, str]) -> str:
        if not self.hitl_handler.is_enabled() or not ApprovalPolicy.requires_approval(name):
            return super().execute(name, args)

        arguments_json = json.dumps(dict(args), ensure_ascii=False)
        request = ApprovalRequest.of(name, arguments_json, None)
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
