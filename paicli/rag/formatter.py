"""Formatting helpers for code search results."""

from __future__ import annotations

from collections import Counter

from paicli.rag.models import SearchResult


class SearchResultFormatter:
    @classmethod
    def format_for_tool(cls, query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"查询: {query}\n未找到相关代码块。"

        files = Counter(result.chunk.file_path for result in results)
        summary = [
            f"查询: {query}",
            f"最相关入口: {results[0].chunk.name}",
            "主要文件: " + ", ".join(file_path for file_path, _ in files.most_common(3)),
            "排序说明: 综合语义相似度、关键词命中、chunk 类型加权和双重命中奖励。",
            "",
            "相关代码:",
        ]
        body: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            body.extend(
                [
                    f"{index}. [{chunk.type.value}] {chunk.name} "
                    f"({chunk.file_path}:{chunk.start_line}-{chunk.end_line}, score={result.similarity:.3f})",
                    "```",
                    chunk.content,
                    "```",
                ]
            )
        return "\n".join(summary + body)

    @classmethod
    def format_for_cli(cls, query: str, results: list[SearchResult]) -> str:
        if not results:
            return f"🔎 检索: {query}\n未找到相关代码块。"

        lines = [
            f"🔎 检索: {query}",
            f"📋 找到 {len(results)} 个相关代码块:",
            "",
            "搜索摘要:",
            f"- 最相关的入口是 {results[0].chunk.name}，位于 {results[0].chunk.file_path}。",
            "- 排序综合参考了关键词命中、语义相似度和代码块类型。",
            "",
        ]
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            lines.extend(
                [
                    f"{index}. [{chunk.type.value}:{chunk.name}] (相似度: {result.similarity:.3f}) {chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
                    cls._preview(chunk.content),
                    "",
                ]
            )
        return "\n".join(lines)

    @classmethod
    def _preview(cls, content: str, max_chars: int = 260) -> str:
        if len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + "..."
