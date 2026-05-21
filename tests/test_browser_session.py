import unittest

from paicli.mcp import McpConfig, McpServerConfig, McpServerManager
from paicli.mcp.browser_session import BrowserMode, BrowserSession


class FakeHitlHandler:
    def __init__(self):
        self.cleared_servers = []

    def clear_approved_all_for_server(self, server_name):
        self.cleared_servers.append(server_name)


class BrowserSessionTest(unittest.TestCase):
    def test_restart_with_args_updates_runtime_config_in_memory(self) -> None:
        manager = McpServerManager(
            McpConfig(
                [
                    McpServerConfig(
                        name="chrome-devtools",
                        transport="stdio",
                        command="npx",
                        args=["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
                    )
                ]
            ),
            client_factory=lambda _server: _FakeClient(),
        )

        result = manager.restart_with_args(
            "chrome-devtools",
            ["-y", "chrome-devtools-mcp@latest", "--autoConnect", "--channel=stable"],
        )

        runtime = manager.runtimes()[0]
        self.assertIn("当前状态: ready", result)
        self.assertEqual(runtime.status, "ready")
        self.assertEqual(
            runtime.config.args,
            ["-y", "chrome-devtools-mcp@latest", "--autoConnect", "--channel=stable"],
        )

    def test_switch_to_shared_uses_auto_connect_and_clears_server_approvals(self) -> None:
        manager = McpServerManager(
            McpConfig(
                [
                    McpServerConfig(
                        name="chrome-devtools",
                        transport="stdio",
                        command="npx",
                        args=["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
                    )
                ]
            ),
            client_factory=lambda _server: _FakeClient(),
        )
        hitl = FakeHitlHandler()
        session = BrowserSession(manager, hitl_handler=hitl)

        result = session.switch_to_shared()

        self.assertEqual(session.mode, BrowserMode.SHARED)
        self.assertIn("--autoConnect", manager.runtimes()[0].config.args)
        self.assertEqual(hitl.cleared_servers, ["chrome-devtools"])
        self.assertIn("shared", result)

    def test_disconnect_returns_to_isolated_mode(self) -> None:
        manager = McpServerManager(
            McpConfig(
                [
                    McpServerConfig(
                        name="chrome-devtools",
                        transport="stdio",
                        command="npx",
                        args=["-y", "chrome-devtools-mcp@latest", "--autoConnect"],
                    )
                ]
            ),
            client_factory=lambda _server: _FakeClient(),
        )
        session = BrowserSession(manager)
        session.switch_to_shared()

        result = session.disconnect()

        self.assertEqual(session.mode, BrowserMode.ISOLATED)
        self.assertEqual(
            manager.runtimes()[0].config.args,
            ["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
        )
        self.assertIn("isolated", result)

    def test_failed_shared_switch_restores_isolated_runtime(self) -> None:
        def factory(server):
            if "--autoConnect" in (server.args or []):
                raise RuntimeError("Chrome remote debugging is not enabled")
            return _FakeClient()

        manager = McpServerManager(
            McpConfig(
                [
                    McpServerConfig(
                        name="chrome-devtools",
                        transport="stdio",
                        command="npx",
                        args=["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
                    )
                ]
            ),
            client_factory=factory,
        )
        session = BrowserSession(manager)

        result = session.switch_to_shared()

        runtime = manager.runtimes()[0]
        self.assertEqual(session.mode, BrowserMode.ISOLATED)
        self.assertEqual(runtime.status, "ready")
        self.assertEqual(runtime.config.args, ["-y", "chrome-devtools-mcp@latest", "--isolated=true"])
        self.assertIn("连接失败", result)


class _FakeClient:
    def list_tools(self):
        return []

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
