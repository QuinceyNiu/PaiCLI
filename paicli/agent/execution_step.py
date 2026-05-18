"""Execution step model for Multi-Agent orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class StepType(str, Enum):
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    COMMAND = "COMMAND"
    ANALYSIS = "ANALYSIS"
    VERIFICATION = "VERIFICATION"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ExecutionStep:
    id: str
    description: str
    type: StepType
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    review: str | None = None
    attempts: int = 0
    start_time: int = 0
    end_time: int = 0

    def mark_started(self) -> None:
        self.status = StepStatus.RUNNING
        self.start_time = _now_ms()

    def mark_completed(self, result: str, review: str | None = None) -> None:
        self.status = StepStatus.COMPLETED
        self.result = result
        self.review = review
        self.error = None
        self.end_time = _now_ms()

    def mark_failed(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self.end_time = _now_ms()

    def mark_skipped(self, error: str) -> None:
        self.status = StepStatus.SKIPPED
        self.error = error
        self.end_time = _now_ms()

    def is_executable(self, all_steps: Mapping[str, "ExecutionStep"]) -> bool:
        if self.status != StepStatus.PENDING:
            return False
        return all(
            all_steps.get(dependency_id) is not None
            and all_steps[dependency_id].status == StepStatus.COMPLETED
            for dependency_id in self.dependencies
        )
