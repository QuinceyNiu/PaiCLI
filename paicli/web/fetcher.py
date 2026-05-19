"""HTTP fetcher with bounded response reads."""

from __future__ import annotations

import re

import httpx

from paicli.web.models import RawResponse
from paicli.web.network_policy import NetworkPolicy


class WebFetcher:
    USER_AGENT = "Mozilla/5.0 (compatible; paicli-web-fetch/1.0)"

    def __init__(
        self,
        policy: NetworkPolicy | None = None,
        client: httpx.Client | None = None,
        max_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.policy = policy or NetworkPolicy()
        self.client = client
        self.max_bytes = max_bytes

    def fetch(self, url: str) -> RawResponse:
        policy_error = self.policy.validate(url)
        if policy_error:
            raise ValueError(policy_error)

        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        try:
            with client.stream("GET", url, headers={"User-Agent": self.USER_AGENT}) as response:
                chunks: list[bytes] = []
                total = 0
                truncated = False
                for chunk in response.iter_bytes(chunk_size=8192):
                    remaining = self.max_bytes - total
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(chunk) > remaining:
                        chunks.append(chunk[:remaining])
                        total += remaining
                        truncated = True
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                body = b"".join(chunks)
                content_type = response.headers.get("content-type", "")
                charset = _charset_from_content_type(content_type) or response.encoding or "utf-8"
                text = body.decode(charset, errors="replace")
                return RawResponse(
                    url=str(response.url),
                    status_code=response.status_code,
                    text=text,
                    truncated=truncated,
                    content_type=content_type,
                )
        finally:
            if owns_client:
                client.close()


def _charset_from_content_type(content_type: str) -> str | None:
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip("\"'")
