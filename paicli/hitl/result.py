"""Approval decisions returned by HITL handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApprovalDecision(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_ALL = "APPROVED_ALL"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ApprovalResult:
    decision: ApprovalDecision
    modified_arguments: str | None = None
    reason: str | None = None

    @classmethod
    def approve(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.APPROVED)

    @classmethod
    def approve_all(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.APPROVED_ALL)

    @classmethod
    def reject(cls, reason: str | None = None) -> "ApprovalResult":
        return cls(ApprovalDecision.REJECTED, reason=reason)

    @classmethod
    def modify(cls, modified_arguments: str) -> "ApprovalResult":
        return cls(ApprovalDecision.MODIFIED, modified_arguments=modified_arguments)

    @classmethod
    def skip(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.SKIPPED)

    def is_rejected(self) -> bool:
        return self.decision == ApprovalDecision.REJECTED

    def is_skipped(self) -> bool:
        return self.decision == ApprovalDecision.SKIPPED

    def effective_arguments(self, original_arguments: str) -> str:
        if (
            self.decision == ApprovalDecision.MODIFIED
            and self.modified_arguments is not None
            and self.modified_arguments.strip()
        ):
            return self.modified_arguments
        return original_arguments
