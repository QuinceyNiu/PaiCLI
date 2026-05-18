import unittest

from paicli.hitl import ApprovalDecision, ApprovalResult


class ApprovalResultTest(unittest.TestCase):
    def test_effective_arguments_uses_original_for_plain_approval(self) -> None:
        result = ApprovalResult.approve()

        self.assertEqual(result.decision, ApprovalDecision.APPROVED)
        self.assertEqual(result.effective_arguments('{"path":"a.txt"}'), '{"path":"a.txt"}')

    def test_effective_arguments_uses_modified_arguments_when_present(self) -> None:
        result = ApprovalResult.modify('{"path":"safe.txt"}')

        self.assertEqual(result.decision, ApprovalDecision.MODIFIED)
        self.assertEqual(result.effective_arguments('{"path":"unsafe.txt"}'), '{"path":"safe.txt"}')

    def test_effective_arguments_falls_back_when_modified_arguments_are_blank(self) -> None:
        result = ApprovalResult.modify("  ")

        self.assertEqual(result.effective_arguments('{"path":"original.txt"}'), '{"path":"original.txt"}')

    def test_rejected_and_skipped_helpers(self) -> None:
        rejected = ApprovalResult.reject("路径不对")
        skipped = ApprovalResult.skip()

        self.assertTrue(rejected.is_rejected())
        self.assertEqual(rejected.reason, "路径不对")
        self.assertTrue(skipped.is_skipped())


if __name__ == "__main__":
    unittest.main()
