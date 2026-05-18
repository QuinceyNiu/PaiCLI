"""Roles used by PaiCli Multi-Agent mode."""

from __future__ import annotations

from enum import Enum


class AgentRole(str, Enum):
    PLANNER = "PLANNER"
    WORKER = "WORKER"
    REVIEWER = "REVIEWER"

    @property
    def display_name(self) -> str:
        return {
            AgentRole.PLANNER: "规划者",
            AgentRole.WORKER: "执行者",
            AgentRole.REVIEWER: "检查者",
        }[self]

    @property
    def description(self) -> str:
        return {
            AgentRole.PLANNER: "负责分析用户任务，制定执行计划，将复杂任务拆解为可执行的子任务",
            AgentRole.WORKER: "负责执行具体任务步骤，调用工具完成文件操作、命令执行等操作",
            AgentRole.REVIEWER: "负责检查执行结果的质量和正确性，提供改进建议",
        }[self]
