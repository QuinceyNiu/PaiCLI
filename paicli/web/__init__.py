"""Web search and fetch tools for PaiCli."""

from paicli.web.fetcher import WebFetcher
from paicli.web.html_extractor import HtmlExtractor
from paicli.web.models import FetchResult, RawResponse, SearchResult
from paicli.web.network_policy import NetworkPolicy
from paicli.web.search_provider import (
    SearchProvider,
    SearxngSearchProvider,
    SerpApiSearchProvider,
    ZhipuSearchProvider,
)
from paicli.web.search_provider_factory import SearchProviderFactory
from paicli.web.tool_provider import WebToolProvider

__all__ = [
    "FetchResult",
    "HtmlExtractor",
    "NetworkPolicy",
    "RawResponse",
    "SearchProvider",
    "SearchProviderFactory",
    "SearchResult",
    "SearxngSearchProvider",
    "SerpApiSearchProvider",
    "WebFetcher",
    "WebToolProvider",
    "ZhipuSearchProvider",
]
