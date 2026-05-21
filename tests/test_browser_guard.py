import io
import json
import tempfile
import unittest
from pathlib import Path

from paicli.mcp.browser_session import BrowserMode, BrowserSession
from paicli.tool.hitl_tool_registry import BrowserGuard, SensitivePagePolicy


class FakeSession(BrowserSession):
    def __init__(self):
        self.mode = BrowserMode.SHARED
        self.current_url = ""
        self.agent_opened_tabs = set()

    def record_opened_tab(self, page_id):
        self.agent_opened_tabs.add(page_id)

    def forget_opened_tab(self, page_id):
        self.agent_opened_tabs.discard(page_id)

    def is_agent_opened_tab(self, page_id):
        return page_id in self.agent_opened_tabs


class BrowserGuardTest(unittest.TestCase):
    def test_sensitive_policy_matches_defaults_and_custom_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patterns = Path(tmp) / "sensitive_patterns.txt"
            patterns.write_text("# internal\n*://admin.example.com/*\n", encoding="utf-8")
            policy = SensitivePagePolicy(custom_path=patterns)

            self.assertEqual(
                policy.match("https://github.com/settings/profile"),
                "*://github.com/settings/*",
            )
            self.assertEqual(
                policy.match("https://admin.example.com/users"),
                "*://admin.example.com/*",
            )
            self.assertIsNone(policy.match("https://github.com/owner/repo"))

    def test_shared_mode_blocks_closing_user_owned_tab(self) -> None:
        session = FakeSession()
        guard = BrowserGuard(session, SensitivePagePolicy(patterns=[]))

        result = guard.check("mcp__chrome_devtools__close_page", {"pageIdx": "2"})

        self.assertFalse(result.allowed)
        self.assertIn("拒绝关闭非 PaiCLI 创建的标签页", result.message)

    def test_shared_mode_allows_closing_agent_opened_tab(self) -> None:
        session = FakeSession()
        session.record_opened_tab("2")
        guard = BrowserGuard(session, SensitivePagePolicy(patterns=[]))

        result = guard.check("mcp__chrome_devtools__close_page", {"pageIdx": "2"})

        self.assertTrue(result.allowed)

    def test_records_new_page_id_after_execution(self) -> None:
        session = FakeSession()
        guard = BrowserGuard(session, SensitivePagePolicy(patterns=[]))

        guard.apply_after_execution(
            "mcp__chrome_devtools__new_page",
            {"url": "https://example.com"},
            "Opened new page with id 7 for https://example.com",
        )

        self.assertTrue(session.is_agent_opened_tab("7"))
        self.assertEqual(session.current_url, "https://example.com")

    def test_sensitive_write_requires_single_approval_request(self) -> None:
        session = FakeSession()
        session.current_url = "https://github.com/settings/profile"
        guard = BrowserGuard(session, SensitivePagePolicy())

        request = guard.approval_request(
            "mcp__chrome_devtools__click",
            json.dumps({"uid": "button-1"}),
        )

        self.assertIsNotNone(request)
        self.assertFalse(request.allow_approve_all)
        self.assertIn("敏感页面", request.suggestion or "")
        self.assertIn("github.com/settings", request.caller_context or "")


if __name__ == "__main__":
    unittest.main()
