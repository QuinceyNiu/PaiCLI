"""One-shot Skill body injection buffer."""

from __future__ import annotations

from collections import OrderedDict


class SkillContextBuffer:
    def __init__(self, max_skills: int = 3) -> None:
        self.max_skills = max_skills
        self._items: OrderedDict[str, str] = OrderedDict()

    def push(self, name: str, body: str) -> None:
        cleaned_name = name.strip()
        if not cleaned_name:
            return
        if cleaned_name in self._items:
            del self._items[cleaned_name]
        self._items[cleaned_name] = body
        while len(self._items) > self.max_skills:
            self._items.popitem(last=False)

    def drain(self) -> str:
        if not self._items:
            return ""
        blocks = [
            f"## 已加载 Skill：{name}\n{body.strip()}"
            for name, body in self._items.items()
        ]
        self._items.clear()
        return "\n\n".join(blocks)

    def clear(self) -> None:
        self._items.clear()
