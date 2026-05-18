"""Orchestrator for PaiCli Multi-Agent mode."""

from __future__ import annotations

import json
import re
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Callable, Optional

from paicli.agent.agent_message import AgentMessage, AgentMessageType
from paicli.agent.agent_role import AgentRole
from paicli.agent.execution_step import ExecutionStep, StepStatus, StepType
from paicli.agent.sub_agent import SubAgent
from paicli.llm.glm_client import GLMClient
from paicli.memory import MemoryManager
from paicli.tool.tool_registry import ToolRegistry


MAX_RETRIES_PER_STEP = 2
DEPENDENCY_RESULT_PREVIEW_CHARS = 500


class AgentOrchestrator:
    def __init__(
        self,
        api_key: str,
        llm_client: Optional[GLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        base_dir: str | Path | None = None,
        on_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.llm_client = llm_client or GLMClient(api_key)
        self.tool_registry = tool_registry or ToolRegistry(base_dir=base_dir)
        self.memory_manager = memory_manager or MemoryManager()
        if self.memory_manager.context_compressor.llm_client is None:
            self.memory_manager.context_compressor.llm_client = self.llm_client
        self.on_event = on_event
        self.planner = SubAgent("planner", AgentRole.PLANNER, self.llm_client, self.tool_registry)
        self.workers = [
            SubAgent("worker-1", AgentRole.WORKER, self.llm_client, self.tool_registry),
            SubAgent("worker-2", AgentRole.WORKER, self.llm_client, self.tool_registry),
        ]
        self.reviewer = SubAgent("reviewer", AgentRole.REVIEWER, self.llm_client, self.tool_registry)

    def run(self, user_input: str) -> str:
        self.memory_manager.add_user_message(user_input)
        self._emit("📋 第一阶段：规划")
        self._emit("🧑‍💼 规划者正在分析任务...")
        memory_context = self.memory_manager.build_context_for_query(user_input, max_tokens=500)
        plan_content = user_input if not memory_context else f"{memory_context}\n\n用户任务：{user_input}"
        plan_result = self.planner.execute(AgentMessage.user_task(plan_content))
        if plan_result.type == AgentMessageType.ERROR:
            return f"规划失败: {plan_result.content}"

        try:
            steps = self.parse_plan(plan_result.content)
        except ValueError as exc:
            return f"规划解析失败: {exc}"

        self._emit(self.format_execution_plan(steps))
        self._emit("⚡ 第二阶段：执行")
        worker_pool: Queue[SubAgent] = Queue()
        for worker in self.workers:
            worker_pool.put(worker)

        while True:
            executable = self.get_executable_steps(steps)
            if not executable:
                break
            self.run_batch_parallel(executable, steps, worker_pool)
            if any(step.status == StepStatus.FAILED for step in executable):
                self.mark_blocked_steps_skipped(steps)
                break

        self.mark_blocked_steps_skipped(steps)
        final_result = self.build_final_result(steps)
        self.memory_manager.add_assistant_message(final_result)
        self.memory_manager.extract_and_save_facts()
        return final_result

    def parse_plan(self, plan_content: str) -> list[ExecutionStep]:
        data = json.loads(self._clean_json(plan_content))
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("执行计划缺少 steps 数组")

        steps: list[ExecutionStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise ValueError("steps 中每一项都必须是对象")
            try:
                step_type = StepType(str(raw_step.get("type", "ANALYSIS")))
            except ValueError:
                step_type = StepType.ANALYSIS
            dependencies = raw_step.get("dependencies") or []
            if not isinstance(dependencies, list):
                raise ValueError("dependencies 必须是数组")
            steps.append(
                ExecutionStep(
                    id=str(raw_step["id"]),
                    description=str(raw_step["description"]),
                    type=step_type,
                    dependencies=[str(item) for item in dependencies],
                )
            )
        return steps

    def get_executable_steps(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        step_map = {step.id: step for step in steps}
        return [step for step in steps if step.is_executable(step_map)]

    def run_batch_parallel(
        self,
        batch: list[ExecutionStep],
        all_steps: list[ExecutionStep],
        worker_pool: Queue[SubAgent],
    ) -> None:
        if len(batch) == 1:
            worker = worker_pool.get()
            try:
                self._call_run_step(batch[0], all_steps, worker, emit=None, reviewer=self.reviewer)
            finally:
                worker.clear_history()
                worker_pool.put(worker)
            return

        buffers: dict[str, list[str]] = {step.id: [] for step in batch}
        with ThreadPoolExecutor(max_workers=min(len(batch), len(self.workers))) as executor:
            futures = []
            for step in batch:
                futures.append(
                    executor.submit(
                        self._run_step_from_pool,
                        step,
                        all_steps,
                        worker_pool,
                        buffers[step.id].append,
                    )
                )
            for future in as_completed(futures):
                future.result()
        for step in batch:
            for message in buffers[step.id]:
                self._emit(message)

    def run_step(
        self,
        step: ExecutionStep,
        all_steps: list[ExecutionStep],
        worker: SubAgent,
        emit: Callable[[str], None] | None = None,
        reviewer: SubAgent | None = None,
    ) -> None:
        emit = emit or self._emit
        reviewer = reviewer or self.reviewer
        step.mark_started()
        context = self.build_step_context(all_steps, step)
        result = ""
        review_content = ""
        approved = False

        for attempt in range(MAX_RETRIES_PER_STEP + 1):
            step.attempts = attempt + 1
            emit(f"🛠️ {worker.agent_id} 执行步骤 [{step.id}]: {step.description}")
            task_content = self._build_worker_task(step, context, review_content if attempt > 0 else "")
            result_message = worker.execute(
                AgentMessage("orchestrator", AgentRole.WORKER, task_content, AgentMessageType.TASK)
            )
            result = result_message.content
            worker.clear_history()

            emit(f"🔍 reviewer 正在审查步骤 [{step.id}] 的结果...")
            review_message = reviewer.review(step.description, result)
            review_content = review_message.content
            approved = self.parse_review_approval(review_content)
            reviewer.clear_history()
            if approved:
                emit(f"✅ 步骤 [{step.id}] 审查通过")
                step.mark_completed(result, review_content)
                return

            if attempt < MAX_RETRIES_PER_STEP:
                emit(f"♻️ 步骤 [{step.id}] 审查未通过，准备第 {attempt + 1} 次重试")

        step.mark_failed(f"审查未通过: {self.extract_review_feedback(review_content) or result}")

    def parse_review_approval(self, review_content: str | None) -> bool:
        if not review_content:
            return False
        try:
            root = json.loads(self._clean_json(review_content))
            if not isinstance(root, dict) or "approved" not in root or root["approved"] is None:
                return False
            return bool(root.get("approved", False))
        except Exception:
            lower = review_content.lower()
            has_negative = "未通过" in lower or "不通过" in lower
            has_positive = "通过" in lower or "合格" in lower
            if has_negative:
                return False
            if not has_positive:
                return False
            return True

    def extract_review_feedback(self, review_content: str | None) -> str:
        if not review_content:
            return ""
        try:
            root = json.loads(self._clean_json(review_content))
        except Exception:
            return review_content
        if not isinstance(root, dict):
            return review_content
        parts = []
        for key, label in (("summary", "摘要"), ("issues", "问题"), ("suggestions", "建议")):
            value = root.get(key)
            if isinstance(value, list):
                parts.append(f"{label}: " + "；".join(str(item) for item in value))
            elif value:
                parts.append(f"{label}: {value}")
        return "\n".join(parts)

    def build_step_context(self, steps: list[ExecutionStep], current_step: ExecutionStep) -> str:
        lines = ["总任务上下文："]
        for step in steps:
            if step.status == StepStatus.COMPLETED and step.id in current_step.dependencies:
                result = step.result or ""
                preview = (
                    result[:DEPENDENCY_RESULT_PREVIEW_CHARS] + "..."
                    if len(result) > DEPENDENCY_RESULT_PREVIEW_CHARS
                    else result
                )
                lines.append(f"已完成的依赖步骤 [{step.id}]: {step.description}")
                lines.append(f"结果：{preview}")
        return "\n".join(lines)

    def mark_blocked_steps_skipped(self, steps: list[ExecutionStep]) -> None:
        step_map = {step.id: step for step in steps}
        changed = True
        while changed:
            changed = False
            for step in steps:
                if step.status != StepStatus.PENDING:
                    continue
                if any(
                    step_map.get(dependency_id) is None
                    or step_map[dependency_id].status in {StepStatus.FAILED, StepStatus.SKIPPED}
                    for dependency_id in step.dependencies
                ):
                    step.mark_skipped("依赖步骤失败或被跳过")
                    changed = True

    def format_execution_plan(self, steps: list[ExecutionStep]) -> str:
        lines = ["📋 执行计划"]
        for step in steps:
            deps = ", ".join(step.dependencies) if step.dependencies else "无"
            lines.append(f"  ⏳ [{step.id}] {step.description} (依赖: {deps})")
        return "\n".join(lines)

    def build_final_result(self, steps: list[ExecutionStep]) -> str:
        lines = ["🤝 Multi-Agent 协作结果"]
        for step in steps:
            icon = {
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
                StepStatus.RUNNING: "▶️",
                StepStatus.PENDING: "⏳",
            }[step.status]
            lines.append(f"{icon} [{step.id}] {step.description}")
            if step.result:
                lines.append(f"   结果: {step.result}")
            if step.error:
                lines.append(f"   问题: {step.error}")
        if all(step.status == StepStatus.COMPLETED for step in steps):
            lines.append("全部步骤审查通过")
        else:
            lines.append("部分步骤未完成，请查看上方问题")
        return "\n".join(lines)

    def _run_step_from_pool(
        self,
        step: ExecutionStep,
        all_steps: list[ExecutionStep],
        worker_pool: Queue[SubAgent],
        emit: Callable[[str], None],
    ) -> None:
        worker = worker_pool.get()
        local_reviewer = SubAgent(
            f"reviewer-{step.id}",
            AgentRole.REVIEWER,
            self.llm_client,
            self.tool_registry,
        )
        try:
            self._call_run_step(step, all_steps, worker, emit=emit, reviewer=local_reviewer)
        finally:
            worker.clear_history()
            worker_pool.put(worker)

    def _call_run_step(
        self,
        step: ExecutionStep,
        all_steps: list[ExecutionStep],
        worker: SubAgent,
        emit: Callable[[str], None] | None,
        reviewer: SubAgent,
    ) -> None:
        parameters = inspect.signature(self.run_step).parameters
        if "reviewer" in parameters:
            self.run_step(step, all_steps, worker, emit=emit, reviewer=reviewer)
        else:
            self.run_step(step, all_steps, worker, emit=emit)

    def _build_worker_task(self, step: ExecutionStep, context: str, review_content: str) -> str:
        lines = [
            context,
            "",
            f"当前步骤 [{step.id}] ({step.type.value}): {step.description}",
            "请调用必要工具完成这个步骤，并汇报执行结果。",
        ]
        feedback = self.extract_review_feedback(review_content)
        if feedback:
            lines.extend(["", "之前的执行结果被审查拒绝，原因：", feedback])
        return "\n".join(lines)

    def _clean_json(self, text: str) -> str:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        return cleaned

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)
