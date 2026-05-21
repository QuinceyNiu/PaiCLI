"""Three-layer Skill registry and load_skill tool provider."""

from __future__ import annotations

import json
import shutil
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from paicli.skill.buffer import SkillContextBuffer
from paicli.skill.models import Skill, SkillSource
from paicli.skill.parser import SkillFrontmatterParser
from paicli.tool.tool_registry import RegisteredTool, ToolProvider, create_parameters


DEFAULT_USER_SKILL_DIR = Path.home() / ".paicli" / "skills"
DEFAULT_STATE_PATH = Path.home() / ".paicli" / "skills.json"
DEFAULT_CACHE_DIR = Path.home() / ".paicli" / "skills-cache"
DEFAULT_BUILTIN_DIR = Path(__file__).resolve().parents[1] / "skills" / "builtin"
MAX_SKILL_BODY_CHARS = 5_000
_ACTIVE_CONTEXT = threading.local()


@contextmanager
def active_skill_context(
    buffer: SkillContextBuffer,
    loaded_names: set[str],
) -> Iterator[None]:
    previous_buffer = getattr(_ACTIVE_CONTEXT, "buffer", None)
    previous_loaded = getattr(_ACTIVE_CONTEXT, "loaded_names", None)
    _ACTIVE_CONTEXT.buffer = buffer
    _ACTIVE_CONTEXT.loaded_names = loaded_names
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.buffer = previous_buffer
        _ACTIVE_CONTEXT.loaded_names = previous_loaded


class SkillRegistry:
    def __init__(
        self,
        builtin_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.builtin_dir = Path(builtin_dir or DEFAULT_BUILTIN_DIR).expanduser()
        self.user_dir = Path(user_dir or DEFAULT_USER_SKILL_DIR).expanduser()
        self.project_dir = Path(project_dir or (Path.cwd() / ".paicli" / "skills")).expanduser()
        self.state_path = Path(state_path or DEFAULT_STATE_PATH).expanduser()
        default_cache_dir = self.state_path.parent / "skills-cache" if state_path is not None else DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir or default_cache_dir).expanduser()
        self.parser = SkillFrontmatterParser()
        self.skills: dict[str, Skill] = {}
        self.disabled: set[str] = set()

    def reload(self) -> None:
        self.disabled = self._load_disabled()
        merged: dict[str, Skill] = {}
        for source, directory in (
            (SkillSource.BUILTIN, self.builtin_dir),
            (SkillSource.USER, self.user_dir),
            (SkillSource.PROJECT, self.project_dir),
        ):
            for skill in self._scan_dir(directory, source):
                merged[skill.name] = skill
        self.skills = {
            name: self._with_enabled(skill, name not in self.disabled)
            for name, skill in sorted(merged.items())
        }
        self._extract_builtin_references()

    def list(self, include_disabled: bool = True) -> list[Skill]:
        skills = list(self.skills.values())
        if include_disabled:
            return skills
        return [skill for skill in skills if skill.enabled]

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name.strip())

    def disable(self, name: str) -> bool:
        if name not in self.skills:
            return False
        self.disabled.add(name)
        self._save_disabled()
        self.skills[name] = self._with_enabled(self.skills[name], False)
        return True

    def enable(self, name: str) -> bool:
        if name not in self.skills:
            return False
        self.disabled.discard(name)
        self._save_disabled()
        self.skills[name] = self._with_enabled(self.skills[name], True)
        return True

    def format_index(self) -> str:
        enabled = self.list(include_disabled=False)
        if not enabled:
            return ""
        lines = [
            "## 可用 Skills（按需调用 load_skill 加载完整指引）",
            "",
        ]
        for skill in enabled:
            lines.append(f"- **{skill.name}**: {skill.description}")
        lines.extend(
            [
                "",
                "判断准则：当任务描述匹配某个 skill 的触发场景时，调用 load_skill(name) 加载完整指引，然后按指引执行。",
                "已加载的 skill 会在下一轮以 `## 已加载 Skill` 段落出现在你的 user message 中。",
                "不要重复加载同一 skill；同一会话内一次足够。",
            ]
        )
        return "\n".join(lines)

    def format_summary(self) -> str:
        skills = self.list()
        index_kb = len(self.format_index().encode("utf-8")) / 1024
        enabled_count = len([skill for skill in skills if skill.enabled])
        lines = [f"📚 Skills 加载（{len(skills)} 个）..."]
        for skill in skills:
            status = "✓" if skill.enabled else "○"
            source = skill.source.value
            lines.append(
                f"   {status} {skill.name:<15} {source:<8} description {len(skill.description)} 字符"
            )
        lines.append(f"   {enabled_count}/{len(skills)} 启用，索引段共 {index_kb:.1f}KB")
        return "\n".join(lines)

    def format_list(self) -> str:
        if not self.skills:
            return "暂无 Skill。"
        lines = ["Skills:"]
        for skill in self.list():
            status = "●" if skill.enabled else "○"
            version = skill.version or "-"
            lines.append(f"{status} {skill.name:<15} {skill.source.value:<8} v{version}")
        return "\n".join(lines)

    def tool_provider(
        self,
        buffer: SkillContextBuffer,
        loaded_names: set[str] | None = None,
    ) -> ToolProvider:
        return _SkillToolProvider(self, buffer, loaded_names if loaded_names is not None else set())

    def _scan_dir(self, directory: Path, source: SkillSource) -> Iterable[Skill]:
        if not directory.exists():
            return []
        skills: list[Skill] = []
        for skill_file in sorted(directory.glob("*/SKILL.md")):
            skill = self._load_skill_file(skill_file, source)
            if skill is not None:
                skills.append(skill)
        return skills

    def _load_skill_file(self, path: Path, source: SkillSource) -> Skill | None:
        document = path.read_text(encoding="utf-8")
        parsed = self.parser.parse(document, skill_name=path.parent.name)
        name = str(parsed.metadata.get("name") or "").strip()
        description = str(parsed.metadata.get("description") or "").strip()
        if not name or not description:
            return None
        tags = parsed.metadata.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        return Skill(
            name=name,
            description=description,
            body=parsed.body,
            document=document,
            path=path,
            source=source,
            version=str(parsed.metadata.get("version") or ""),
            author=str(parsed.metadata.get("author") or ""),
            tags=[str(tag) for tag in tags],
            enabled=name not in self.disabled,
        )

    def _extract_builtin_references(self) -> None:
        for skill in self.list():
            if skill.source != SkillSource.BUILTIN:
                continue
            try:
                self._extract_builtin_skill_references(skill)
            except OSError as exc:
                print(
                    f"⚠️ Skill '{skill.name}' references 解压失败：{exc}",
                    file=sys.stderr,
                )

    def _extract_builtin_skill_references(self, skill: Skill) -> None:
        source_dir = skill.path.parent
        target_dir = self.cache_dir / skill.name
        version = skill.version or "0"
        version_file = target_dir / ".version"
        if version_file.exists() and version_file.read_text(encoding="utf-8") == version:
            return
        if target_dir.exists():
            shutil.rmtree(target_dir)
        references = source_dir / "references"
        if references.exists():
            shutil.copytree(references, target_dir / "references")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text(version, encoding="utf-8")

    def _load_disabled(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        disabled = data.get("disabled") if isinstance(data, dict) else []
        if not isinstance(disabled, list):
            return set()
        return {str(name) for name in disabled}

    def _save_disabled(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"disabled": sorted(self.disabled)}
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _with_enabled(self, skill: Skill, enabled: bool) -> Skill:
        return Skill(
            name=skill.name,
            description=skill.description,
            body=skill.body,
            document=skill.document,
            path=skill.path,
            source=skill.source,
            version=skill.version,
            author=skill.author,
            tags=list(skill.tags),
            enabled=enabled,
        )


class _SkillToolProvider:
    def __init__(
        self,
        registry: SkillRegistry,
        buffer: SkillContextBuffer,
        loaded_names: set[str],
    ) -> None:
        self.registry = registry
        self.buffer = buffer
        self.loaded_names = loaded_names

    def get_tools(self) -> list[RegisteredTool]:
        return [
            RegisteredTool(
                name="load_skill",
                description="按名称加载 Skill 的完整决策指引。只在任务匹配可用 Skill 时调用，同一会话同一 Skill 只需加载一次。",
                parameters=create_parameters(("name", "string", "Skill 名称", True)),
                executor=self._load_skill,
            )
        ]

    def _load_skill(self, args) -> str:
        name = str(args.get("name") or "").strip()
        if not name:
            return "加载 Skill 失败: name 不能为空"
        skill = self.registry.get(name)
        if skill is None:
            return f"加载 Skill 失败: 未找到 skill '{name}'"
        if not skill.enabled:
            return f"加载 Skill 失败: skill '{name}' 已禁用"
        buffer = getattr(_ACTIVE_CONTEXT, "buffer", None) or self.buffer
        loaded_names = getattr(_ACTIVE_CONTEXT, "loaded_names", None) or self.loaded_names
        if name in loaded_names:
            return f"skill '{name}' 已在本会话加载过，无需重复加载"
        body = skill.body[:MAX_SKILL_BODY_CHARS]
        buffer.push(skill.name, body)
        loaded_names.add(name)
        size_kb = len(body.encode("utf-8")) / 1024
        truncated = "，内容已截断" if len(skill.body) > MAX_SKILL_BODY_CHARS else ""
        return f"已加载 skill '{name}' 的完整指引（{size_kb:.1f}KB{truncated}），将在下一轮上下文中体现"
