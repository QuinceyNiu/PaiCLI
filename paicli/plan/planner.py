"""LLM-backed planner for Plan-and-Execute mode."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from paicli.llm.glm_client import ChatResponse, Message
from paicli.plan.execution_plan import ExecutionPlan
from paicli.plan.task import Task, TaskType


PLANNING_PROMPT = """
你是 PaiCLI 的任务规划器，负责把用户的复杂目标拆分成可执行计划。

请按以下JSON格式输出执行计划，不要输出任何解释性文字：
{
  "summary": "任务摘要",
  "tasks": [
    {
      "id": "task_1",
      "description": "任务描述",
      "type": "FILE_READ",
      "dependencies": [],
      "arguments": {"path": "README.md"}
    }
  ]
}

可用任务类型：
- PLANNING: 规划任务，用于分析和决策
- FILE_READ: 读取文件内容，用于获取信息
- FILE_WRITE: 写入文件内容，用于输出结果
- COMMAND: 执行Shell命令，用于编译运行等
- ANALYSIS: 分析结果，用于中间决策
- VERIFICATION: 验证结果，用于检查正确性

规则：
1. 每个任务必须有唯一的id，如 task_1、task_2。
2. dependencies 只列出当前任务依赖的任务 id。
3. 任务应该按推荐执行顺序排列。
4. 任务描述要具体明确，能指导后续执行器行动。
5. 复杂任务拆分为 5-10 个子任务，简单任务可以更少。
6. 依赖关系必须是有向无环图，不能出现循环依赖。
7. FILE_READ 参数使用 {"path": "文件路径"}。
8. FILE_WRITE 参数使用 {"path": "文件路径", "content": "文件内容"}。
9. COMMAND 和 VERIFICATION 参数使用 {"command": "Shell命令"}。
10. 涉及代码理解、定位实现、调用关系或重构影响分析的任务，应在描述中明确要求先使用 search_code 检索代码库上下文。
""".strip()


class PlanningError(Exception):
    """Raised when the planner cannot build a valid execution plan."""


class PlannerLLMClient(Protocol):
    def chat(self, messages: list[Message], tools: Any | None = None) -> ChatResponse:
        """Create a planning response from the LLM."""


class Planner:
    def __init__(self, llm_client: PlannerLLMClient) -> None:
        self.llm_client = llm_client

    def create_plan(self, goal: str) -> ExecutionPlan:
        messages = [
            Message.system(PLANNING_PROMPT),
            Message.user("请为以下任务制定执行计划：\n" + goal),
        ]
        response = self.llm_client.chat(messages, None)
        return self.parse_plan(goal, response.message.content or "")

    def replan(self, failed_plan: ExecutionPlan, failure_reason: str) -> ExecutionPlan:
        context = self._build_replan_context(failed_plan, failure_reason)
        return self.create_plan(context)

    def parse_plan(self, goal: str, plan_json: str) -> ExecutionPlan:
        cleaned = self._clean_plan_json(plan_json)
        try:
            root = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"规划结果不是合法 JSON: {exc}") from exc

        if not isinstance(root, dict):
            raise PlanningError("规划结果必须是 JSON 对象")

        tasks_data = root.get("tasks")
        if not isinstance(tasks_data, list):
            raise PlanningError("规划结果缺少 tasks 数组")

        plan = ExecutionPlan(id=self._generate_plan_id(), goal=goal)
        summary = root.get("summary")
        if isinstance(summary, str) and summary.strip():
            plan.summary = summary.strip()

        id_occurrences: dict[str, list[tuple[int, str]]] = {}
        pending_dependencies: list[tuple[int, Task, list[str]]] = []

        for index, task_data in enumerate(tasks_data, start=1):
            if not isinstance(task_data, dict):
                raise PlanningError("tasks 中的每一项都必须是 JSON 对象")

            original_id = str(task_data.get("id") or f"task_{index}")
            new_id = f"task_{index}"
            id_occurrences.setdefault(original_id, []).append((index, new_id))

            description = str(task_data.get("description") or "").strip()
            if not description:
                raise PlanningError(f"{new_id} 缺少 description")

            task_type = self._parse_task_type(task_data.get("type"), new_id)
            arguments = self._parse_arguments(task_data.get("arguments", {}), new_id)
            task = Task(
                id=new_id,
                description=description,
                type=task_type,
                arguments=arguments,
            )
            plan.add_task(task)

            raw_dependencies = task_data.get("dependencies", [])
            if not isinstance(raw_dependencies, list):
                raise PlanningError(f"{new_id} 的 dependencies 必须是数组")
            pending_dependencies.append((index, task, [str(dep) for dep in raw_dependencies]))

        for task_index, task, original_dependencies in pending_dependencies:
            task.dependencies = [
                self._map_dependency_id(dependency, task_index, id_occurrences)
                for dependency in original_dependencies
                if dependency in id_occurrences
            ]

        if not plan.compute_execution_order():
            raise PlanningError("规划结果存在循环依赖，无法执行")

        return plan

    def _map_dependency_id(
        self,
        dependency: str,
        task_index: int,
        id_occurrences: dict[str, list[tuple[int, str]]],
    ) -> str:
        candidates = id_occurrences[dependency]
        previous_candidates = [
            new_id
            for original_index, new_id in candidates
            if original_index < task_index
        ]
        if previous_candidates:
            return previous_candidates[-1]
        return candidates[0][1]

    def _build_replan_context(self, failed_plan: ExecutionPlan, failure_reason: str) -> str:
        completed_lines = [
            f"- {task.description}: {task.result or ''}".rstrip()
            for task in failed_plan.tasks.values()
            if task.status.value == "COMPLETED"
        ]
        failed_lines = [
            f"- {task.description}: {task.error or ''}".rstrip()
            for task in failed_plan.tasks.values()
            if task.status.value == "FAILED"
        ]

        completed_text = "\n".join(completed_lines) if completed_lines else "- 无"
        failed_text = "\n".join(failed_lines) if failed_lines else "- 无"

        return "\n".join(
            [
                "请基于以下执行进度重新制定后续计划。",
                f"原始目标：{failed_plan.goal}",
                "已完成任务：",
                completed_text,
                "失败任务：",
                failed_text,
                f"失败原因：{failure_reason}",
                "新计划应尽量复用已完成结果，只包含仍需执行的后续任务。",
            ]
        )

    def _parse_task_type(self, raw_type: object, task_id: str) -> TaskType:
        try:
            return TaskType(str(raw_type))
        except ValueError as exc:
            raise PlanningError(f"{task_id} 的任务类型无效: {raw_type}") from exc

    def _parse_arguments(self, raw_arguments: object, task_id: str) -> dict[str, str]:
        if raw_arguments is None:
            return {}
        if not isinstance(raw_arguments, dict):
            raise PlanningError(f"{task_id} 的 arguments 必须是 JSON 对象")
        return {
            str(key): value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in raw_arguments.items()
        }

    def _clean_plan_json(self, plan_json: str) -> str:
        cleaned = plan_json.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json") :].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```") :].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].strip()

        return cleaned

    def _generate_plan_id(self) -> str:
        return f"plan_{int(time.time() * 1000)}"
