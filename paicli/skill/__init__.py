"""Skill loading, indexing, and prompt-injection support for PaiCLI."""

from paicli.skill.buffer import SkillContextBuffer
from paicli.skill.parser import ParsedSkillDocument, SkillFrontmatterParser
from paicli.skill.registry import SkillRegistry
from paicli.skill.registry import active_skill_context
from paicli.skill.models import Skill, SkillSource

__all__ = [
    "ParsedSkillDocument",
    "Skill",
    "SkillContextBuffer",
    "SkillFrontmatterParser",
    "SkillRegistry",
    "SkillSource",
    "active_skill_context",
]
