import unittest

from paicli.hitl import ApprovalPolicy


class ApprovalPolicyTest(unittest.TestCase):
    def test_requires_approval_for_mutating_tools(self) -> None:
        self.assertTrue(ApprovalPolicy.requires_approval("write_file"))
        self.assertTrue(ApprovalPolicy.requires_approval("execute_command"))
        self.assertTrue(ApprovalPolicy.requires_approval("create_project"))

        self.assertFalse(ApprovalPolicy.requires_approval("read_file"))
        self.assertFalse(ApprovalPolicy.requires_approval("list_dir"))
        self.assertFalse(ApprovalPolicy.requires_approval("search_code"))

    def test_danger_levels_are_stable_static_rules(self) -> None:
        self.assertEqual(ApprovalPolicy.get_danger_level("execute_command"), "🔴 高危")
        self.assertEqual(ApprovalPolicy.get_danger_level("write_file"), "🟡 中危")
        self.assertEqual(ApprovalPolicy.get_danger_level("create_project"), "🟡 中危")
        self.assertEqual(ApprovalPolicy.get_danger_level("read_file"), "🟢 安全")

    def test_risk_descriptions_explain_why_approval_is_needed(self) -> None:
        self.assertIn("Shell 命令", ApprovalPolicy.get_risk_description("execute_command"))
        self.assertIn("覆盖文件", ApprovalPolicy.get_risk_description("write_file"))
        self.assertIn("创建新目录和文件", ApprovalPolicy.get_risk_description("create_project"))
        self.assertIn("只读", ApprovalPolicy.get_risk_description("read_file"))


if __name__ == "__main__":
    unittest.main()
