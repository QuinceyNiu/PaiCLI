"""Transport implementations for MCP JSON-RPC messages."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from typing import Any, Callable, Protocol

import httpx

from paicli.mcp.config import McpServerConfig


MessageHandler = Callable[[dict[str, Any]], None]


class McpTransport(Protocol):
    def send(
        self,
        message: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC message."""

    def on_receive(self, handler: MessageHandler) -> None:
        """Register a listener for incoming JSON-RPC messages."""

    def close(self) -> None:
        """Close transport resources."""


class StdioTransport:
    def __init__(self, config: McpServerConfig) -> None:
        if not config.command:
            raise ValueError("stdio MCP server 缺少 command")
        self.config = config
        self.stderr_lines: deque[str] = deque(maxlen=200)
        self._handlers: list[MessageHandler] = []
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [config.command, *config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **config.env},
        )
        threading.Thread(
            target=self._read_stdout,
            name=f"paicli-mcp-stdout-{config.name}",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            name=f"paicli-mcp-stderr-{config.name}",
            daemon=True,
        ).start()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def send(
        self,
        message: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if "id" not in message:
            self._write_message(message)
            return None

        event = threading.Event()
        response: dict[str, Any] = {}

        def capture(payload: dict[str, Any]) -> None:
            if payload.get("id") == message.get("id"):
                response.update(payload)
                event.set()

        self.on_receive(capture)
        self._write_message(message)
        timeout = (timeout_seconds or 60.0) + 1.0
        if not event.wait(timeout):
            raise TimeoutError(f"MCP stdio 请求超时: {message.get('method')}")
        return response

    def on_receive(self, handler: MessageHandler) -> None:
        self._handlers.append(handler)

    def logs(self) -> list[str]:
        return list(self.stderr_lines)

    def close(self) -> None:
        self.graceful_close()

    def graceful_close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            pass

        process.terminate()
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def _write_message(self, message: dict[str, Any]) -> None:
        with self._lock:
            stdin = self._process.stdin
            if stdin is None:
                raise RuntimeError("MCP stdio stdin 不可用")
            stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            stdin.flush()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        for line in stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            for handler in list(self._handlers):
                handler(message)

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            self.stderr_lines.append(line.rstrip("\n"))


class StreamableHttpTransport:
    def __init__(
        self,
        config: McpServerConfig,
        protocol_version: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not config.url:
            raise ValueError("HTTP MCP server 缺少 url")
        self.config = config
        self.protocol_version = protocol_version
        self.session_id: str | None = None
        self._handlers: list[MessageHandler] = []
        self._client = http_client or httpx.Client(timeout=60.0)
        self._owns_client = http_client is None

    def send(
        self,
        message: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        response = self._client.post(
            self.config.url or "",
            headers=self._headers(),
            json=message,
            timeout=(timeout_seconds or 60.0) + 1.0,
        )
        self._capture_session(response)
        response.raise_for_status()
        payload = _decode_http_response(response)
        if payload is not None:
            for handler in list(self._handlers):
                handler(payload)
        return payload

    def on_receive(self, handler: MessageHandler) -> None:
        self._handlers.append(handler)

    def close(self) -> None:
        if self.session_id and self.config.url:
            try:
                self._client.delete(self.config.url, headers=self._headers())
            except Exception:
                pass
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
            **self.config.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _capture_session(self, response: httpx.Response) -> None:
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self.session_id = session_id


def _decode_http_response(response: httpx.Response) -> dict[str, Any] | None:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        if not response.content:
            return None
        return response.json()

    data_lines: list[str] = []
    for raw_line in response.text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue
        if line == "" and data_lines:
            payload = "\n".join(data_lines)
            data_lines.clear()
            if payload and payload != "[DONE]":
                return json.loads(payload)
    if data_lines:
        payload = "\n".join(data_lines)
        if payload and payload != "[DONE]":
            return json.loads(payload)
    return None
