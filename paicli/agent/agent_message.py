"""Messages routed between the orchestrator and sub agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from paicli.agent.agent_role import AgentRole


class AgentMessageType(str, Enum):
    TASK = "TASK"
    RESULT = "RESULT"
    FEEDBACK = "FEEDBACK"
    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AgentMessage:
    from_agent: str
    from_role: AgentRole
    content: str
    type: AgentMessageType

    @classmethod
    def user_task(cls, content: str) -> "AgentMessage":
        return cls(
            from_agent="user",
            from_role=AgentRole.PLANNER,
            content=content,
            type=AgentMessageType.TASK,
        )
