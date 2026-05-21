"""MCP server configuration loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MCP_CONFIG_PATH = Path.home() / ".paicli" / "mcp.json"
PROJECT_MCP_CONFIG_PATH = Path(".paicli") / "mcp.json"
DEFAULT_CHROME_DEVTOOLS_SERVER = {
    "command": "npx",
    "args": ["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
}
DEFAULT_MCP_TEMPLATE = {
    "mcpServers": {
        "chrome-devtools": DEFAULT_CHROME_DEVTOOLS_SERVER,
    }
}


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", list(self.args or []))
        object.__setattr__(self, "env", dict(self.env or {}))
        object.__setattr__(self, "headers", dict(self.headers or {}))


@dataclass(frozen=True)
class McpConfig:
    servers: list[McpServerConfig]

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_MCP_CONFIG_PATH,
        project_dir: str | Path = ".",
    ) -> "McpConfig":
        config_path = Path(path).expanduser()
        if not config_path.exists():
            return cls([])

        return cls.from_mapping(
            json.loads(config_path.read_text(encoding="utf-8")),
            project_dir=project_dir,
        )

    @classmethod
    def load_merged(
        cls,
        user_path: str | Path = DEFAULT_MCP_CONFIG_PATH,
        project_path: str | Path = PROJECT_MCP_CONFIG_PATH,
        project_dir: str | Path = ".",
    ) -> "McpConfig":
        merged: dict[str, Any] = {}
        for path in [Path(user_path).expanduser(), Path(project_path).expanduser()]:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers") if isinstance(data, Mapping) else None
            if isinstance(servers, Mapping):
                merged.update(dict(servers))
        return cls.from_mapping({"mcpServers": merged}, project_dir=project_dir)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        project_dir: str | Path = ".",
    ) -> "McpConfig":
        if not isinstance(data, Mapping):
            raise ValueError("MCP 配置必须是 JSON 对象")

        raw_servers = data.get("mcpServers") or {}
        if not isinstance(raw_servers, Mapping):
            raise ValueError("mcpServers 必须是 JSON 对象")

        project = Path(project_dir).expanduser().resolve()
        servers: list[McpServerConfig] = []
        for name, value in raw_servers.items():
            try:
                servers.append(_parse_server(str(name), value, project))
            except Exception as exc:
                servers.append(
                    McpServerConfig(
                        name=str(name),
                        transport="stdio",
                        enabled=True,
                        env={"PAICLI_MCP_CONFIG_ERROR": str(exc)},
                    )
                )
        return cls(servers)

    def server_names(self) -> list[str]:
        return [server.name for server in self.servers]


def ensure_default_mcp_config(path: str | Path = DEFAULT_MCP_CONFIG_PATH) -> str:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(DEFAULT_MCP_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return f"🔌 已创建默认 MCP 配置: {config_path}"
        except Exception as exc:
            return f"🔌 默认 MCP 配置创建失败: {exc}"

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"🔌 MCP 配置读取失败: {exc}"
    servers = data.get("mcpServers") if isinstance(data, Mapping) else None
    if isinstance(servers, Mapping) and "chrome-devtools" not in servers:
        return "🔌 MCP 配置缺少 chrome-devtools；可手动添加 Google 官方 Chrome DevTools MCP。"
    return ""


def _parse_server(name: str, value: Any, project_dir: Path) -> McpServerConfig:
    if not isinstance(value, Mapping):
        raise ValueError(f"MCP server {name} 必须是 JSON 对象")

    raw_type = str(value.get("type") or value.get("transport") or "stdio").lower()
    if "url" in value and "command" not in value and "type" not in value and "transport" not in value:
        raw_type = "http"
    transport = "http" if raw_type in {"http", "streamable_http", "streamable-http"} else "stdio"
    variables = _variables(project_dir)
    args = [_expand_placeholders(str(arg), variables) for arg in value.get("args") or []]
    env = {
        str(key): _expand_placeholders(str(env_value), variables)
        for key, env_value in (value.get("env") or {}).items()
    }
    headers = {
        str(key): _expand_placeholders(str(header_value), variables)
        for key, header_value in (value.get("headers") or {}).items()
    }
    command = value.get("command")
    url = value.get("url")

    if transport == "stdio" and not command:
        raise ValueError(f"MCP server {name} 缺少 command")
    if transport == "http" and not url:
        raise ValueError(f"MCP server {name} 缺少 url")

    return McpServerConfig(
        name=name,
        transport=transport,
        command=str(command) if command else None,
        args=args,
        env=env,
        url=str(url) if url else None,
        headers=headers,
        enabled=bool(value.get("enabled", True)),
    )


def _variables(project_dir: Path) -> dict[str, str]:
    return {
        **{key: str(value) for key, value in os.environ.items()},
        "PROJECT_DIR": str(project_dir),
        "HOME": str(Path.home()),
    }


def _expand_placeholders(value: str, variables: Mapping[str, str]) -> str:
    result = value
    for name in _placeholder_names(value):
        if name not in variables:
            raise ValueError(f"环境变量 {name} 未设置")
        result = result.replace("${" + name + "}", variables[name])
    return result


def _placeholder_names(value: str) -> list[str]:
    import re

    return re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
