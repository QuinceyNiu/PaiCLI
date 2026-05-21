"""Data models for PaiCLI skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SkillSource(str, Enum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    document: str
    path: Path
    source: SkillSource
    version: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def size_kb(self) -> float:
        return len(self.body.encode("utf-8")) / 1024
