"""Execution plan model and dependency ordering."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from paicli.plan.task import Task, TaskStatus


TASK_STATUS_ICONS = {
    TaskStatus.PENDING: "⏳",
    TaskStatus.RUNNING: "▶️",
    TaskStatus.COMPLETED: "✅",
    TaskStatus.FAILED: "❌",
    TaskStatus.SKIPPED: "⏭️",
}


class PlanStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ExecutionPlan:
    id: str
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED
    summary: str | None = None
    start_time: int = 0
    end_time: int = 0

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def compute_execution_order(self) -> bool:
        self.execution_order.clear()
        self._rebuild_dependents()
        visited: set[str] = set()
        visiting: set[str] = set()

        for task in self.tasks.values():
            if task.id not in visited:
                if not self._topological_sort(task, visited, visiting):
                    self.execution_order.clear()
                    return False

        return True

    def mark_started(self) -> None:
        self.status = PlanStatus.RUNNING
        self.start_time = _now_ms()

    def mark_completed(self, summary: str | None = None) -> None:
        self.status = PlanStatus.COMPLETED
        self.summary = summary
        self.end_time = _now_ms()

    def mark_failed(self, summary: str | None = None) -> None:
        self.status = PlanStatus.FAILED
        self.summary = summary
        self.end_time = _now_ms()

    def mark_cancelled(self, summary: str | None = None) -> None:
        self.status = PlanStatus.CANCELLED
        self.summary = summary
        self.end_time = _now_ms()

    def has_failed(self) -> bool:
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def visualize(self) -> str:
        lines = [
            f"执行计划: {self.goal}",
            f"状态: {self.status.value}",
        ]
        if self.summary:
            lines.append(f"摘要: {self.summary}")

        for index, task_id in enumerate(self.execution_order, start=1):
            task = self.tasks[task_id]
            dependencies = ", ".join(task.dependencies) if task.dependencies else "无"
            icon = TASK_STATUS_ICONS[task.status]
            lines.append(
                f"{index}. {icon} {task.id} [{task.status.value}] "
                f"({task.type.value}) - {task.description} | 依赖: {dependencies}"
            )

        return "\n".join(lines)

    def _topological_sort(
        self,
        task: Task,
        visited: set[str],
        visiting: set[str],
    ) -> bool:
        task_id = task.id
        if task_id in visiting:
            return False
        if task_id in visited:
            return True

        visiting.add(task_id)

        for dependency_id in task.dependencies:
            dependency = self.tasks.get(dependency_id)
            if dependency is not None:
                if not self._topological_sort(dependency, visited, visiting):
                    return False

        visiting.remove(task_id)
        visited.add(task_id)
        self.execution_order.append(task_id)
        return True

    def _rebuild_dependents(self) -> None:
        for task in self.tasks.values():
            task.dependents.clear()

        for task in self.tasks.values():
            for dependency_id in task.dependencies:
                dependency = self.tasks.get(dependency_id)
                if dependency is not None and task.id not in dependency.dependents:
                    dependency.dependents.append(task.id)
