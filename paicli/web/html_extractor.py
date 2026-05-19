"""Extract readable Markdown from static HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

from paicli.web.models import FetchResult


EMPTY_BODY_MESSAGE = "未提取到正文。可能是 JS 渲染或防爬墙；本期范围内不再重试。"
NOISE_TAGS = {"script", "style", "nav", "aside", "footer", "header", "form", "iframe", "noscript"}
NOISE_KEYWORDS = ("ads", "advert", "banner", "sidebar", "comment", "footer", "header", "nav")
BLOCK_TAGS = {"article", "main", "section", "div", "body"}


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_Node | str"] = field(default_factory=list)

    def text(self) -> str:
        parts: list[str] = []
        for child in self.children:
            parts.append(child if isinstance(child, str) else child.text())
        return re.sub(r"\s+", " ", "".join(parts)).strip()


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag.lower(), {key.lower(): value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag.lower() not in {"br", "img", "meta", "link", "input", "hr"}:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == lowered:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)


class HtmlExtractor:
    def extract(self, url: str, html_text: str, truncated: bool = False) -> FetchResult:
        root = self._parse(html_text)
        title = self._title(root)
        main = self._find_semantic_main(root) or self._best_scored_block(root)
        markdown = self._markdown(main, url).strip() if main is not None else ""
        if not markdown:
            markdown = EMPTY_BODY_MESSAGE
        return FetchResult(url=url, title=title, markdown=markdown, truncated=truncated)

    def _parse(self, html_text: str) -> _Node:
        parser = _TreeBuilder()
        parser.feed(html_text)
        return parser.root

    def _title(self, root: _Node) -> str:
        for node in self._walk(root):
            if node.tag == "title":
                return node.text()
        return ""

    def _find_semantic_main(self, root: _Node) -> _Node | None:
        for node in self._walk(root):
            if self._is_noise(node):
                continue
            if node.tag in {"article", "main"} or node.attrs.get("role", "").lower() == "main":
                return node
        return None

    def _best_scored_block(self, root: _Node) -> _Node | None:
        best: tuple[float, _Node | None] = (0.0, None)
        for node in self._walk(root):
            if node.tag not in BLOCK_TAGS or self._is_noise(node):
                continue
            score = self._score(node)
            if score > best[0]:
                best = (score, node)
        return best[1]

    def _score(self, node: _Node) -> float:
        text = node.text()
        text_len = len(text)
        if text_len < 80:
            return 0.0
        link_len = sum(child.text().__len__() for child in self._walk(node) if child.tag == "a")
        penalty = min((link_len / text_len) * 2.0, 1.0) if text_len else 1.0
        return text_len * (1.0 - penalty)

    def _markdown(self, node: _Node, base_url: str) -> str:
        if self._is_noise(node):
            return ""
        tag = node.tag
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            return f"{'#' * level} {self._inline(node, base_url).strip()}\n\n"
        if tag == "p":
            return f"{self._inline(node, base_url).strip()}\n\n"
        if tag == "br":
            return "\n"
        if tag == "pre":
            code = node.text().strip()
            return f"```\n{code}\n```\n\n" if code else ""
        if tag == "ul":
            return "".join(f"- {self._inline(child, base_url).strip()}\n" for child in self._element_children(node, "li")) + "\n"
        if tag == "ol":
            return "".join(f"{index}. {self._inline(child, base_url).strip()}\n" for index, child in enumerate(self._element_children(node, "li"), 1)) + "\n"
        if tag == "table":
            return self._table(node, base_url)
        return "".join(self._markdown(child, base_url) if isinstance(child, _Node) else "" for child in node.children)

    def _inline(self, node: _Node, base_url: str) -> str:
        parts: list[str] = []
        for child in node.children:
            if isinstance(child, str):
                parts.append(html.unescape(child))
                continue
            if self._is_noise(child):
                continue
            text = self._inline(child, base_url)
            if child.tag in {"strong", "b"} and text.strip():
                parts.append(f"**{text.strip()}**")
            elif child.tag in {"em", "i"} and text.strip():
                parts.append(f"*{text.strip()}*")
            elif child.tag == "code" and text.strip():
                parts.append(f"`{text.strip()}`")
            elif child.tag == "a" and text.strip():
                href = child.attrs.get("href", "").strip()
                parts.append(f"[{text.strip()}]({urljoin(base_url, href)})" if href else text)
            elif child.tag == "br":
                parts.append("\n")
            else:
                parts.append(text)
        return re.sub(r"[ \t]+", " ", "".join(parts))

    def _table(self, node: _Node, base_url: str) -> str:
        rows: list[list[str]] = []
        for row in self._walk(node):
            if row.tag != "tr":
                continue
            cells = [
                self._inline(cell, base_url).strip().replace("|", "\\|")
                for cell in row.children
                if isinstance(cell, _Node) and cell.tag in {"th", "td"}
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        widths = max(len(row) for row in rows)
        rows = [row + [""] * (widths - len(row)) for row in rows]
        lines = ["| " + " | ".join(rows[0]) + " |"]
        lines.append("| " + " | ".join("---" for _ in range(widths)) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        return "\n".join(lines) + "\n\n"

    def _walk(self, node: _Node):
        yield node
        for child in node.children:
            if isinstance(child, _Node):
                yield from self._walk(child)

    def _element_children(self, node: _Node, tag: str) -> list[_Node]:
        return [child for child in node.children if isinstance(child, _Node) and child.tag == tag]

    def _is_noise(self, node: _Node) -> bool:
        if node.tag in NOISE_TAGS:
            return True
        marker = " ".join([node.attrs.get("class", ""), node.attrs.get("id", "")]).lower()
        return any(keyword in marker for keyword in NOISE_KEYWORDS)
