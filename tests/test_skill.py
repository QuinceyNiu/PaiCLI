import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from paicli.skill import (
    SkillContextBuffer,
    SkillFrontmatterParser,
    SkillRegistry,
    SkillSource,
)
from paicli.tool.tool_registry import ToolRegistry


class SkillFrontmatterParserTest(unittest.TestCase):
    def test_parse_single_line_multiline_description_and_tags(self) -> None:
        document = """---
name: web-access
description: |
  所有联网操作必须通过此 skill 处理，
  包括搜索、网页抓取、登录后操作
version: "1.0.0"
author: PaiCLI
tags: [web, browser, search]
---

# web-access Skill
"""

        parsed = SkillFrontmatterParser().parse(document, skill_name="web-access")

        self.assertEqual(parsed.metadata["name"], "web-access")
        self.assertEqual(
            parsed.metadata["description"],
            "所有联网操作必须通过此 skill 处理，\n包括搜索、网页抓取、登录后操作",
        )
        self.assertEqual(parsed.metadata["version"], "1.0.0")
        self.assertEqual(parsed.metadata["author"], "PaiCLI")
        self.assertEqual(parsed.metadata["tags"], ["web", "browser", "search"])
        self.assertEqual(parsed.body.strip(), "# web-access Skill")

    def test_unsupported_frontmatter_field_warns_and_skips_field(self) -> None:
        document = """---
name: broken
description: ok
metadata: {nested: object}
---

body
"""
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            parsed = SkillFrontmatterParser().parse(document, skill_name="broken")

        self.assertNotIn("metadata", parsed.metadata)
        self.assertIn("Skill 'broken' frontmatter 解析警告", stderr.getvalue())
        self.assertIn("不支持的语法", stderr.getvalue())


class SkillRegistryTest(unittest.TestCase):
    def test_scan_merges_builtin_user_project_with_later_sources_overriding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = root / "builtin"
            user = root / "user"
            project = root / "project"
            _write_skill(builtin / "web-access" / "SKILL.md", "web-access", "builtin desc", "1.0.0")
            _write_skill(user / "web-access" / "SKILL.md", "web-access", "user desc", "9.9.9")
            _write_skill(project / "code-review" / "SKILL.md", "code-review", "review desc", "1.0.0")

            registry = SkillRegistry(
                builtin_dir=builtin,
                user_dir=user,
                project_dir=project,
                state_path=root / "skills.json",
            )
            registry.reload()

            web = registry.get("web-access")
            review = registry.get("code-review")
            self.assertIsNotNone(web)
            self.assertEqual(web.description, "user desc")
            self.assertEqual(web.version, "9.9.9")
            self.assertEqual(web.source, SkillSource.USER)
            self.assertEqual(review.source, SkillSource.PROJECT)

    def test_disabled_state_is_persisted_and_hides_skill_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = root / "builtin"
            _write_skill(builtin / "web-access" / "SKILL.md", "web-access", "builtin desc", "1.0.0")
            registry = SkillRegistry(
                builtin_dir=builtin,
                user_dir=root / "user",
                project_dir=root / "project",
                state_path=root / "skills.json",
            )
            registry.reload()

            registry.disable("web-access")
            reloaded = SkillRegistry(
                builtin_dir=builtin,
                user_dir=root / "user",
                project_dir=root / "project",
                state_path=root / "skills.json",
            )
            reloaded.reload()

            self.assertFalse(reloaded.get("web-access").enabled)
            self.assertNotIn("web-access", reloaded.format_index())
            self.assertEqual(json.loads((root / "skills.json").read_text())["disabled"], ["web-access"])

    def test_builtin_extractor_copies_references_once_per_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builtin = root / "builtin" / "web-access"
            cache = root / "cache"
            _write_skill(builtin / "SKILL.md", "web-access", "builtin desc", "1.2.3")
            reference = builtin / "references" / "site-patterns" / "mp.weixin.qq.com.md"
            reference.parent.mkdir(parents=True)
            reference.write_text("微信经验", encoding="utf-8")

            registry = SkillRegistry(
                builtin_dir=root / "builtin",
                user_dir=root / "user",
                project_dir=root / "project",
                cache_dir=cache,
                state_path=root / "skills.json",
            )
            registry.reload()
            cached = cache / "web-access" / "references" / "site-patterns" / "mp.weixin.qq.com.md"
            cached.write_text("用户缓存修改", encoding="utf-8")

            registry.reload()

            self.assertEqual((cache / "web-access" / ".version").read_text(encoding="utf-8"), "1.2.3")
            self.assertEqual(cached.read_text(encoding="utf-8"), "用户缓存修改")


class SkillContextBufferTest(unittest.TestCase):
    def test_buffer_keeps_recent_three_replaces_same_name_and_drains_once(self) -> None:
        buffer = SkillContextBuffer(max_skills=3)

        buffer.push("one", "body 1")
        buffer.push("two", "body 2")
        buffer.push("three", "body 3")
        buffer.push("two", "body 2 new")
        buffer.push("four", "body 4")

        injected = buffer.drain()
        self.assertNotIn("body 1", injected)
        self.assertIn("## 已加载 Skill：three", injected)
        self.assertIn("## 已加载 Skill：two", injected)
        self.assertIn("body 2 new", injected)
        self.assertIn("## 已加载 Skill：four", injected)
        self.assertEqual(buffer.drain(), "")


class SkillToolTest(unittest.TestCase):
    def test_load_skill_tool_pushes_body_to_buffer_without_returning_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root / "builtin" / "web-access" / "SKILL.md", "web-access", "联网指引", "1.0.0", body="完整正文")
            registry = SkillRegistry(
                builtin_dir=root / "builtin",
                user_dir=root / "user",
                project_dir=root / "project",
                state_path=root / "skills.json",
            )
            registry.reload()
            buffer = SkillContextBuffer()
            tools = ToolRegistry(providers=[registry.tool_provider(buffer)])

            result = tools.execute("load_skill", {"name": "web-access"})

            self.assertIn("已加载 skill 'web-access'", result)
            self.assertNotIn("完整正文", result)
            self.assertIn("完整正文", buffer.drain())

    def test_load_skill_refuses_disabled_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root / "builtin" / "web-access" / "SKILL.md", "web-access", "联网指引", "1.0.0")
            registry = SkillRegistry(
                builtin_dir=root / "builtin",
                user_dir=root / "user",
                project_dir=root / "project",
                state_path=root / "skills.json",
            )
            registry.reload()
            registry.disable("web-access")
            buffer = SkillContextBuffer()
            tools = ToolRegistry(providers=[registry.tool_provider(buffer)])

            result = tools.execute("load_skill", {"name": "web-access"})

            self.assertIn("已禁用", result)
            self.assertEqual(buffer.drain(), "")


def _write_skill(path: Path, name: str, description: str, version: str, body: str = "body") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {name}
description: {description}
version: "{version}"
---

{body}
""",
        encoding="utf-8",
    )
