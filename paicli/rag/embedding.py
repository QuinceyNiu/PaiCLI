"""Embedding client abstraction used by the local RAG index."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Protocol


MAX_INPUT_CHARS = 2_000
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text:latest"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "embedding-3"
HTTP_TIMEOUT = (30.0, 120.0)


class EmbeddingTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: tuple[float, float],
    ) -> dict[str, Any]:
        """POST JSON and return decoded response data."""


class HttpxEmbeddingTransport:
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: tuple[float, float],
    ) -> dict[str, Any]:
        import httpx

        client_timeout = httpx.Timeout(
            timeout=None,
            connect=timeout[0],
            read=timeout[1],
            write=timeout[1],
            pool=timeout[0],
        )
        with httpx.Client(timeout=client_timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


class EmbeddingClient:
    TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_input_chars: int = MAX_INPUT_CHARS,
        transport: EmbeddingTransport | None = None,
        dimensions: int = 64,
    ) -> None:
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER") or DEFAULT_PROVIDER).lower()
        self.model = model or os.getenv("EMBEDDING_MODEL") or self._default_model(self.provider)
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or self._default_base_url(self.provider)
        ).rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or ""
        self.max_input_chars = max_input_chars
        self.transport = transport or HttpxEmbeddingTransport()
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        input_text = text[: self.max_input_chars]
        if self.provider == "local":
            return self._embed_local(input_text)
        if self.provider in {"openai", "zhipu", "glm"}:
            return self._embed_openai_compatible(input_text)
        return self._embed_ollama(input_text)

    def _embed_ollama(self, text: str) -> list[float]:
        try:
            data = self.transport.post_json(
                url=f"{self.base_url}/api/embeddings",
                payload={"model": self.model, "prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=HTTP_TIMEOUT,
            )
            return self._coerce_vector(data.get("embedding"))
        except Exception:
            data = self.transport.post_json(
                url=f"{self.base_url}/api/embed",
                payload={"model": self.model, "input": text},
                headers={"Content-Type": "application/json"},
                timeout=HTTP_TIMEOUT,
            )
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                return self._coerce_vector(embeddings[0])
            return self._coerce_vector(data.get("embedding"))

    def _embed_openai_compatible(self, text: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self.transport.post_json(
            url=f"{self.base_url}/embeddings",
            payload={"model": self.model, "input": text},
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        items = data.get("data") or []
        if not items:
            raise ValueError("Embedding response did not include data[0].embedding")
        return self._coerce_vector(items[0].get("embedding"))

    def _embed_local(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self.TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _coerce_vector(self, raw_vector: Any) -> list[float]:
        if not isinstance(raw_vector, list):
            raise ValueError("Embedding response did not include a vector")
        return [float(value) for value in raw_vector]

    def _default_model(self, provider: str) -> str:
        if provider == "ollama":
            return DEFAULT_OLLAMA_MODEL
        return DEFAULT_OPENAI_COMPATIBLE_MODEL

    def _default_base_url(self, provider: str) -> str:
        if provider in {"glm", "zhipu"}:
            return DEFAULT_GLM_BASE_URL
        if provider == "openai":
            return DEFAULT_OPENAI_BASE_URL
        return DEFAULT_OLLAMA_BASE_URL
