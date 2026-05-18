"""Plan-and-Execute agent that combines planning with local tool execution."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol

from paicli.agent.agent import Agent
from paicli.llm.glm_client import GLMClient
from paicli.memory import MemoryManager
from paicli.plan.execution_plan import ExecutionPlan
from paicli.plan.planner import Planner, PlanningError
from paicli.plan.task import Task, TaskStatus, TaskType
from paicli.tool.tool_registry import ToolRegistry


class PlannerLike(Protocol):
    def create_plan(self, goal: str) -> ExecutionPlan:
        """Create an execution plan for a goal."""


class ReactAgentLike(Protocol):
    def run(self, user_input: str) -> str:
        """Run a simple task with the ReAct agent."""


class PlanExecuteAgent:
    ACTION_KEYWORDS = ("创建", "写", "读", "执行", "然后", "接着")

    def __init__(
        self,
        api_key: str,
        llm_client: Optional[GLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[PlannerLike] = None,
        react_agent: Optional[ReactAgentLike] = None,
        base_dir: str | Path | None = None,
        on_plan_update: Optional[Callable[[str], None]] = None,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.llm_client = llm_client or GLMClient(api_key)
        self.tool_registry = tool_registry or ToolRegistry(base_dir=base_dir)
        self.on_plan_update = on_plan_update
        self.memory_manager = memory_manager or MemoryManager()
        if memory_manager is None and self.memory_manager.context_compressor.llm_client is None:
            self.memory_manager.context_compressor.llm_client = self.llm_client
        self.planner = planner or Planner(self.llm_client)
        self.react_agent = react_agent or Agent(
            api_key=api_key,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
        )

    def run(self, user_input: str) -> str:
        if not self.should_plan(user_input):
            return self.react_agent.run(user_input)

        try:
            plan = self.create_plan(user_input)
        except PlanningError as exc:
            return f"规划失败: {exc}"

        return self.execute_plan(plan)

    def create_plan(self, goal: str) -> ExecutionPlan:
        return self.planner.create_plan(goal)

    def execute_plan(self, plan: ExecutionPlan) -> str:
        plan.mark_started()
        self._emit_plan_update(plan)
        for task_id in plan.execution_order:
            task = plan.tasks[task_id]
            if not task.is_executable(plan.tasks):
                task.mark_skipped("依赖任务未完成")
                self._emit_plan_update(plan)
                continue

            self._execute_task(task, plan)
            if task.status == TaskStatus.FAILED:
                self._skip_blocked_tasks(plan)
                plan.mark_failed(task.error)
                self._emit_plan_update(plan)
                return self._build_result(plan)

        plan.mark_completed("全部任务执行完成")
        self.memory_manager.extract_and_save_facts()
        self._emit_plan_update(plan)
        return self._build_result(plan)

    def should_plan(self, user_input: str) -> bool:
        action_count = sum(1 for keyword in self.ACTION_KEYWORDS if keyword in user_input)
        return action_count >= 3 or len(user_input) > 50

    def _execute_task(self, task: Task, plan: ExecutionPlan) -> None:
        task.mark_started()
        self._emit_plan_update(plan)
        self.memory_manager.build_context_for_query(task.description, limit=5)
        tool_name = self._tool_name_for_task(task)
        if tool_name is None:
            task.mark_completed(f"已完成: {task.description}")
            self._emit_plan_update(plan)
            return

        result = self.tool_registry.execute(tool_name, task.arguments)
        if self._is_failure_result(result):
            task.mark_failed(result)
            self.memory_manager.add_tool_result(tool_name, result)
            self._emit_plan_update(plan)
            return

        task.mark_completed(result)
        self.memory_manager.add_tool_result(tool_name, result)
        self._emit_plan_update(plan)

    def _tool_name_for_task(self, task: Task) -> str | None:
        if task.type == TaskType.FILE_READ:
            return "read_file"
        if task.type == TaskType.FILE_WRITE:
            return "write_file"
        if task.type in {TaskType.COMMAND, TaskType.VERIFICATION}:
            return "execute_command"
        return None

    def _is_failure_result(self, result: str) -> bool:
        failure_markers = (
            "失败",
            "未知工具",
            "exit code: 1",
            "exit code: 2",
            "exit code: 126",
            "exit code: 127",
        )
        return any(marker in result for marker in failure_markers)

    def _skip_blocked_tasks(self, plan: ExecutionPlan) -> None:
        for task_id in plan.execution_order:
            task = plan.tasks[task_id]
            if task.status == TaskStatus.PENDING and not task.is_executable(plan.tasks):
                task.mark_skipped("依赖任务失败")
                self._emit_plan_update(plan)

    def _emit_plan_update(self, plan: ExecutionPlan) -> None:
        if self.on_plan_update is not None:
            self.on_plan_update(plan.visualize())

    def _build_result(self, plan: ExecutionPlan) -> str:
        lines = [plan.visualize(), ""]
        if plan.status.value == "COMPLETED":
            lines.append("全部任务执行完成")
        elif plan.status.value == "FAILED":
            lines.append("任务执行失败")
        else:
            lines.append(f"计划状态: {plan.status.value}")

        for task_id in plan.execution_order:
            task = plan.tasks[task_id]
            if task.result:
                lines.append(f"- {task.id}: {task.result}")
            if task.error:
                lines.append(f"- {task.id}: {task.error}")

        return "\n".join(lines)
