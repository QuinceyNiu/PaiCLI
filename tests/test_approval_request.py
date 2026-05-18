import json
import unittest

from paicli.hitl import ApprovalRequest, display_width


class ApprovalRequestTest(unittest.TestCase):
    def test_from_tool_arguments_adds_policy_metadata(self) -> None:
        request = ApprovalRequest.of(
            "write_file",
            json.dumps({"path": "配置.json", "content": "hello"}, ensure_ascii=False),
            "确认路径是否正确",
        )

        self.assertEqual(request.tool_name, "write_file")
        self.assertEqual(request.danger_level, "🟡 中危")
        self.assertIn("覆盖文件", request.risk_description)
        self.assertEqual(request.suggestion, "确认路径是否正确")

    def test_display_width_counts_cjk_and_emoji_as_wide(self) -> None:
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("配置"), 4)
        self.assertEqual(display_width("⚠️"), 2)

    def test_to_display_text_formats_json_fields_and_truncates_long_values(self) -> None:
        request = ApprovalRequest.of(
            "write_file",
            json.dumps({"path": "配置.json", "content": "x" * 130}, ensure_ascii=False),
            None,
        )

        text = request.to_display_text()

        self.assertIn("需要审批", text)
        self.assertIn("工具: write_file", text)
        self.assertIn('path: "配置.json"', text)
        self.assertIn("(130 字符)", text)
        self.assertIn("...", text)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("┌"))
        self.assertTrue(lines[-1].startswith("└"))


if __name__ == "__main__":
    unittest.main()
