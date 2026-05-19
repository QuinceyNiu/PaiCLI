"""Network policy for web fetching."""

from __future__ import annotations

import ipaddress
import socket
import time
from collections import deque
from urllib.parse import urlparse


class NetworkPolicy:
    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_times: deque[float] = deque()

    def validate(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "只允许访问 http 和 https URL"
        if not parsed.hostname:
            return "URL 缺少 host"

        host_error = self._check_host(parsed.hostname)
        if host_error:
            return host_error

        now = time.monotonic()
        while self._request_times and now - self._request_times[0] > self.window_seconds:
            self._request_times.popleft()
        if len(self._request_times) >= self.max_requests:
            return "请求过于频繁，请稍后再试。"
        self._request_times.append(now)
        return None

    def _check_host(self, host: str) -> str | None:
        lower = host.strip().lower().rstrip(".")
        if lower == "localhost":
            return "禁止访问 localhost"
        if lower.endswith(".localhost"):
            return "禁止访问 localhost"

        direct_error = self._check_ip_literal(lower)
        if direct_error:
            return direct_error

        try:
            addresses = socket.getaddrinfo(lower, None)
        except OSError:
            return None
        for address in addresses:
            candidate = address[4][0]
            ip_error = self._check_ip_literal(candidate)
            if ip_error:
                return ip_error
        return None

    def _check_ip_literal(self, value: str) -> str | None:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return None
        if ip.is_loopback:
            return "禁止访问环回地址"
        if ip.is_private:
            return "禁止访问站内地址"
        if ip.is_link_local:
            return "禁止访问链路本地地址"
        if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return "禁止访问非公网地址"
        return None
