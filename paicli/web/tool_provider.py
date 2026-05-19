"""Tool provider that exposes web search and fetch to PaiCli agents."""

from __future__ import annotations

import threading
from typing import Callable

from paicli.tool.tool_registry import RegisteredTool, create_parameters
from paicli.web.fetcher import WebFetcher
from paicli.web.html_extractor import HtmlExtractor
from paicli.web.models import FetchResult, SearchResult
from paicli.web.search_provider import SearchProvider
from paicli.web.search_provider_factory import SearchProviderFactory


DEFAULT_FETCH_MAX_CHARS = 8_000


class WebToolProvider:
    def __init__(
        self,
        search_provider_factory: Callable[[], SearchProvider] | None = None,
        fetcher_factory: Callable[[], WebFetcher] | None = None,
        extractor_factory: Callable[[], HtmlExtractor] | None = None,
    ) -> None:
        self._search_provider_factory = search_provider_factory or SearchProviderFactory.create
        self._fetcher_factory = fetcher_factory or WebFetcher
        self._extractor_factory = extractor_factory or HtmlExtractor
        self._search_provider: SearchProvider | None = None
        self._fetcher: WebFetcher | None = None
        self._extractor: HtmlExtractor | None = None
        self._lock = threading.Lock()

    def get_tools(self) -> list[RegisteredTool]:
        return [
            RegisteredTool(
                name="web_search",
                description=(
                    "搜索互联网，获取实时信息（最新版本、官方文档、技术资讯、新闻等）。"
                    "当你不知道应该访问哪个网页时使用。"
                ),
                parameters=create_parameters(
                    ("query", "string", "搜索关键词", True),
                    ("top_k", "integer", "返回结果数量（默认5，最多10）", False),
                ),
                executor=self._web_search,
            ),
            RegisteredTool(
                name="web_fetch",
                description=(
                    "抓取指定 URL，提取正文并转成 Markdown。适用静态/SSR 页面；"
                    "JS 渲染或防爬站可能返回空正文，并会给出明确提示。"
                ),
                parameters=create_parameters(
                    ("url", "string", "完整 URL", True),
                    ("max_chars", "integer", "最大字符数（默认8000）", False),
                ),
                executor=self._web_fetch,
            ),
        ]

    def _web_search(self, args) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "搜索失败: query 不能为空"
        top_k = _parse_int(args.get("top_k"), 5, 1, 10)
        provider = self._search()
        if not provider.is_ready():
            return provider.unavailable_hint()
        results = provider.search(query, top_k)
        return _format_search_results(provider.name(), query, results)

    def _web_fetch(self, args) -> str:
        url = str(args.get("url") or "").strip()
        if not url:
            return "抓取失败: url 不能为空"
        max_chars = _parse_int(args.get("max_chars"), DEFAULT_FETCH_MAX_CHARS, 500, 50_000)
        raw = self._fetcher_instance().fetch(url)
        result = self._extractor_instance().extract(raw.url, raw.text, raw.truncated)
        return _format_fetch_result(result, max_chars)

    def _search(self) -> SearchProvider:
        if self._search_provider is None:
            with self._lock:
                if self._search_provider is None:
                    self._search_provider = self._search_provider_factory()
        return self._search_provider

    def _fetcher_instance(self) -> WebFetcher:
        if self._fetcher is None:
            with self._lock:
                if self._fetcher is None:
                    self._fetcher = self._fetcher_factory()
        return self._fetcher

    def _extractor_instance(self) -> HtmlExtractor:
        if self._extractor is None:
            with self._lock:
                if self._extractor is None:
                    self._extractor = self._extractor_factory()
        return self._extractor


def _parse_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _format_search_results(provider: str, query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"搜索完成，但没有找到结果。\nProvider: {provider}\nQuery: {query}"
    lines = [f"搜索结果（Provider: {provider}，Query: {query}）:"]
    for index, result in enumerate(results, 1):
        lines.extend(
            [
                f"{index}. {result.title or '(无标题)'}",
                f"   URL: {result.url}",
                f"   摘要: {result.snippet}",
            ]
        )
    return "\n".join(lines)


def _format_fetch_result(result: FetchResult, max_chars: int) -> str:
    body = result.markdown
    truncated_for_tool = len(body) > max_chars
    if truncated_for_tool:
        body = body[:max_chars] + "\n...(内容已截断)"
    lines = [f"URL: {result.url}"]
    if result.title:
        lines.append(f"标题: {result.title}")
    if result.truncated:
        lines.append("提示: 原始响应超过上限，已截断后提取。")
    if truncated_for_tool:
        lines.append(f"提示: 正文超过 {max_chars} 字符，已截断。")
    lines.extend(["", body])
    return "\n".join(lines)
