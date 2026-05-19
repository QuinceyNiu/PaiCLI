"""Factory for selecting the best configured search provider."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from paicli.web.search_provider import (
    SearchProvider,
    SearxngSearchProvider,
    SerpApiSearchProvider,
    ZhipuSearchProvider,
)


class SearchProviderFactory:
    @staticmethod
    def create() -> SearchProvider:
        config = SearchProviderFactory.load_config()
        provider_name = SearchProviderFactory.pick_provider(
            config.get("SEARCH_PROVIDER", ""),
            config.get("GLM_API_KEY", ""),
            config.get("SERPAPI_KEY", "") or config.get("SERPAPI_API_KEY", ""),
            config.get("SEARXNG_URL", ""),
        )
        if provider_name == "serpapi":
            return SerpApiSearchProvider(config.get("SERPAPI_KEY") or config.get("SERPAPI_API_KEY"))
        if provider_name == "searxng":
            return SearxngSearchProvider(config.get("SEARXNG_URL"))
        return ZhipuSearchProvider(
            config.get("GLM_API_KEY"),
            config.get("ZHIPU_SEARCH_ENGINE") or "search_std",
        )

    @staticmethod
    def pick_provider(
        explicit: str | None,
        glm_key: str | None,
        serp_key: str | None,
        searxng_url: str | None,
    ) -> str:
        if explicit and explicit.strip():
            return explicit.strip().lower()
        if glm_key and glm_key.strip():
            return "zhipu"
        if serp_key and serp_key.strip():
            return "serpapi"
        if searxng_url and searxng_url.strip():
            return "searxng"
        return "zhipu"

    @staticmethod
    def load_config() -> dict[str, str]:
        config: dict[str, str] = {}
        for path in (Path.home() / ".env", Path.cwd() / ".env"):
            if path.exists():
                for key, value in dotenv_values(path).items():
                    if value is not None:
                        config[key] = value
        config.update({key: value for key, value in os.environ.items() if value is not None})
        return config
