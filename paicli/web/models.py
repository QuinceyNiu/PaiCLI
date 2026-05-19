"""Shared web search and fetch result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class RawResponse:
    url: str
    status_code: int
    text: str
    truncated: bool = False
    content_type: str = ""


@dataclass(frozen=True)
class FetchResult:
    url: str
    title: str
    markdown: str
    truncated: bool = False
