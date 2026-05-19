"""Search provider protocol and concrete web search providers."""

from __future__ import annotations

from typing import Protocol

import httpx

from paicli.web.models import SearchResult


class SearchProvider(Protocol):
    def name(self) -> str:
        """Return the provider name."""

    def is_ready(self) -> bool:
        """Return whether this provider has enough configuration to run."""

    def unavailable_hint(self) -> str:
        """Return a user-facing setup hint when the provider is unavailable."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search the web and return normalized results."""


class ZhipuSearchProvider:
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/web_search"
    DEFAULT_ENGINE = "search_std"
    ALLOWED_ENGINES = {"search_std", "search_pro", "search_pro_sogou", "search_pro_quark"}

    def __init__(
        self,
        api_key: str | None,
        search_engine: str = DEFAULT_ENGINE,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.search_engine = (
            search_engine if search_engine in self.ALLOWED_ENGINES else self.DEFAULT_ENGINE
        )
        self.client = client

    def name(self) -> str:
        return "zhipu"

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def unavailable_hint(self) -> str:
        return "智谱搜索不可用：请配置 GLM_API_KEY。"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.is_ready():
            return []
        count = max(1, min(top_k, 10))
        payload = {
            "search_engine": self.search_engine,
            "search_query": query,
            "count": count,
            "content_size": "medium",
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        try:
            response = client.post(
                self.ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if owns_client:
                client.close()
        return _parse_results(data.get("search_result") or data.get("results") or [], count)


class SerpApiSearchProvider:
    ENDPOINT = "https://serpapi.com/search.json"

    def __init__(self, api_key: str | None, client: httpx.Client | None = None) -> None:
        self.api_key = (api_key or "").strip()
        self.client = client

    def name(self) -> str:
        return "serpapi"

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def unavailable_hint(self) -> str:
        return "SerpAPI 搜索不可用：请配置 SERPAPI_KEY。"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.is_ready():
            return []
        count = max(1, min(top_k, 10))
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        try:
            response = client.get(
                self.ENDPOINT,
                params={
                    "q": query,
                    "api_key": self.api_key,
                    "num": str(count),
                    "hl": "zh-cn",
                },
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if owns_client:
                client.close()

        results = _parse_results(data.get("organic_results") or [], count)
        if results:
            return results
        answer_box = data.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("title")
        if not answer:
            return []
        return [SearchResult(str(answer_box.get("title") or "Answer Box"), str(answer_box.get("link") or ""), str(answer))]


class SearxngSearchProvider:
    def __init__(self, base_url: str | None, client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.client = client

    def name(self) -> str:
        return "searxng"

    def is_ready(self) -> bool:
        return bool(self.base_url)

    def unavailable_hint(self) -> str:
        return "SearXNG 搜索不可用：请配置 SEARXNG_URL。"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.is_ready():
            return []
        count = max(1, min(top_k, 10))
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        try:
            response = client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "language": "zh-CN"},
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if owns_client:
                client.close()
        return _parse_results(data.get("results") or [], count)


def _parse_results(items: list[dict], count: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in items[:count]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if title or url or snippet:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results
