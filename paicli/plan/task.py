"""Task model used by Plan-and-Execute mode."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class TaskType(str, Enum):
    PLANNING = "PLANNING"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    COMMAND = "COMMAND"
    ANALYSIS = "ANALYSIS"
    VERIFICATION = "VERIFICATION"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Task:
    id: str
    description: str
    type: TaskType
    dependencies: list[str] = field(default_factory=list)
    arguments: dict[str, str] = field(default_factory=dict)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    start_time: int = 0
    end_time: int = 0

    def mark_started(self) -> None:
        self.status = TaskStatus.RUNNING
        self.start_time = _now_ms()

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.error = None
        self.end_time = _now_ms()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.end_time = _now_ms()

    def mark_skipped(self, error: str) -> None:
        self.status = TaskStatus.SKIPPED
        self.error = error
        self.end_time = _now_ms()

    def is_executable(self, all_tasks: Mapping[str, "Task"]) -> bool:
        if self.status != TaskStatus.PENDING:
            return False

        for dependency_id in self.dependencies:
            dependency = all_tasks.get(dependency_id)
            if dependency is None or dependency.status != TaskStatus.COMPLETED:
                return False

        return True
