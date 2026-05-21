import json
import tempfile
import unittest
from pathlib import Path

import httpx

from paicli.mcp import (
    DEFAULT_CHROME_DEVTOOLS_SERVER,
    DEFAULT_MCP_TEMPLATE,
    McpConfig,
    McpHttpClient,
    McpServerConfig,
    McpToolProvider,
    ensure_default_mcp_config,
    normalize_tool_name,
)


class McpConfigTest(unittest.TestCase):
    def test_load_config_expands_project_dir_and_defaults_to_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp.json"
            project_dir = Path(tmp) / "project"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "filesystem": {
                                "command": "npx",
                                "args": [
                                    "-y",
                                    "@modelcontextprotocol/server-filesystem",
                                    "${PROJECT_DIR}",
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = McpConfig.load(config_path, project_dir=project_dir)

            self.assertEqual(len(config.servers), 1)
            server = config.servers[0]
            self.assertEqual(server.name, "filesystem")
            self.assertEqual(server.transport, "stdio")
            self.assertEqual(server.command, "npx")
            self.assertEqual(server.args[-1], str(project_dir.resolve()))

    def test_load_config_reads_http_server_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "zread": {
                                "type": "http",
                                "url": "https://open.bigmodel.cn/api/mcp/zread/mcp",
                                "headers": {"Authorization": "Bearer test"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = McpConfig.load(config_path, project_dir=tmp)

            self.assertEqual(config.servers[0].transport, "http")
            self.assertEqual(config.servers[0].url, "https://open.bigmodel.cn/api/mcp/zread/mcp")
            self.assertEqual(config.servers[0].headers["Authorization"], "Bearer test")

    def test_load_missing_config_returns_empty_config(self) -> None:
        config = McpConfig.load("/missing/mcp.json", project_dir=".")

        self.assertEqual(config.servers, [])

    def test_ensure_default_config_creates_chrome_devtools_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".paicli" / "mcp.json"

            message = ensure_default_mcp_config(config_path)

            data = json.loads(config_path.read_text(encoding="utf-8"))
            server = data["mcpServers"]["chrome-devtools"]
            self.assertEqual(server, DEFAULT_CHROME_DEVTOOLS_SERVER)
            self.assertIn("已创建默认 MCP 配置", message)

    def test_ensure_default_config_warns_without_overwriting_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp.json"
            original = {"mcpServers": {"zread": {"url": "https://example.com/mcp"}}}
            config_path.write_text(json.dumps(original), encoding="utf-8")

            message = ensure_default_mcp_config(config_path)

            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), original)
            self.assertIn("缺少 chrome-devtools", message)


class McpHttpClientTest(unittest.TestCase):
    def test_lists_tools_with_json_rpc_post(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            payload = json.loads(request.content)
            if payload["method"] == "initialize":
                return httpx.Response(200, headers={"Mcp-Session-Id": "session-1"}, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
            if payload["method"] == "notifications/initialized":
                self.assertEqual(request.headers["Mcp-Session-Id"], "session-1")
                return httpx.Response(202)
            if payload["method"] == "tools/list":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "read",
                                    "description": "Read GitHub files",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}},
                                    },
                                }
                            ]
                        },
                    },
                )
            return httpx.Response(500)

        client = McpHttpClient(
            McpServerConfig(
                name="zread",
                transport="http",
                url="https://example.com/mcp",
                headers={"Authorization": "Bearer token"},
            ),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        tools = client.list_tools()

        self.assertEqual(tools[0].name, "read")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer token")
        self.assertEqual(requests[0].headers["MCP-Protocol-Version"], "2025-03-26")
        self.assertEqual(json.loads(requests[0].content)["method"], "initialize")
        self.assertEqual(json.loads(requests[1].content)["method"], "notifications/initialized")
        self.assertEqual(json.loads(requests[2].content)["method"], "tools/list")

    def test_calls_tool_and_returns_structured_content_as_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if payload["method"] == "initialize":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
            if payload["method"] == "notifications/initialized":
                return httpx.Response(202)
            self.assertEqual(payload["method"], "tools/call")
            self.assertEqual(payload["params"]["name"], "read")
            self.assertEqual(payload["params"]["arguments"], {"repo": "itwanger/paicoding"})
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [
                            {"type": "text", "text": "# PaiCoding"},
                            {"type": "json", "json": {"ok": True}},
                        ]
                    },
                },
            )

        client = McpHttpClient(
            McpServerConfig(name="zread", transport="http", url="https://example.com/mcp"),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = client.call_tool("read", {"repo": "itwanger/paicoding"})

        self.assertIn("# PaiCoding", result)
        self.assertIn('"ok": true', result)


class McpToolProviderTest(unittest.TestCase):
    def test_provider_exposes_server_tools_with_namespaced_names(self) -> None:
        class FakeClient:
            def list_tools(self):
                from paicli.mcp import McpTool

                return [
                    McpTool(
                        name="get-repo.structure",
                        description="Show repo tree",
                        input_schema={"type": "object", "properties": {"repo": {"type": "string"}}},
                    )
                ]

            def call_tool(self, name, arguments):
                return f"{name}:{arguments['repo']}"

        config = McpConfig(
            [
                McpServerConfig(name="zread", transport="http", url="https://example.com/mcp")
            ]
        )
        provider = McpToolProvider(config, client_factory=lambda server: FakeClient())

        tools = provider.get_tools()

        self.assertEqual(tools[0].name, "mcp__zread__get_repo_structure")
        self.assertIn("zread.get-repo.structure", tools[0].description)
        self.assertEqual(
            tools[0].executor({"repo": "itwanger/paicoding"}),
            "get-repo.structure:itwanger/paicoding",
        )

    def test_normalize_tool_name_keeps_llm_safe_identifier(self) -> None:
        self.assertEqual(normalize_tool_name("read-file.now"), "read_file_now")


if __name__ == "__main__":
    unittest.main()
