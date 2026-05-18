import tempfile
import unittest
from pathlib import Path

from paicli.hitl import ApprovalRequest, ApprovalResult
from paicli.tool.hitl_tool_registry import HitlToolRegistry


class FakeHitlHandler:
    def __init__(self, enabled=True, result=None):
        self.enabled = enabled
        self.result = result or ApprovalResult.approve()
        self.requests: list[ApprovalRequest] = []

    def is_enabled(self):
        return self.enabled

    def set_enabled(self, enabled):
        self.enabled = enabled

    def request_approval(self, request):
        self.requests.append(request)
        return self.result


class HitlToolRegistryTest(unittest.TestCase):
    def test_dangerous_tool_is_intercepted_when_hitl_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handler = FakeHitlHandler(enabled=True, result=ApprovalResult.approve())
            registry = HitlToolRegistry(handler, base_dir=tmp)

            registry.execute("write_file", {"path": "test.txt", "content": "hello"})

            self.assertEqual(len(handler.requests), 1)
            self.assertEqual(handler.requests[0].tool_name, "write_file")
            self.assertEqual((Path(tmp) / "test.txt").read_text(), "hello")

    def test_safe_tool_bypasses_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("hello", encoding="utf-8")
            handler = FakeHitlHandler(enabled=True)
            registry = HitlToolRegistry(handler, base_dir=tmp)

            result = registry.execute("read_file", {"path": "test.txt"})

            self.assertEqual(handler.requests, [])
            self.assertIn("hello", result)

    def test_disabled_hitl_bypasses_interception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handler = FakeHitlHandler(enabled=False, result=ApprovalResult.reject("不该调用"))
            registry = HitlToolRegistry(handler, base_dir=tmp)

            registry.execute("write_file", {"path": "test.txt", "content": "hello"})

            self.assertEqual(handler.requests, [])
            self.assertEqual((Path(tmp) / "test.txt").read_text(), "hello")

    def test_rejected_tool_returns_reason_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handler = FakeHitlHandler(enabled=True, result=ApprovalResult.reject("路径有误"))
            registry = HitlToolRegistry(handler, base_dir=tmp)

            result = registry.execute("write_file", {"path": "test.txt", "content": "hello"})

            self.assertEqual(result, "[HITL] 操作已被拒绝：路径有误")
            self.assertFalse((Path(tmp) / "test.txt").exists())

    def test_skipped_tool_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handler = FakeHitlHandler(enabled=True, result=ApprovalResult.skip())
            registry = HitlToolRegistry(handler, base_dir=tmp)

            result = registry.execute("write_file", {"path": "test.txt", "content": "hello"})

            self.assertEqual(result, "[HITL] 操作已被跳过")
            self.assertFalse((Path(tmp) / "test.txt").exists())

    def test_modified_arguments_are_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handler = FakeHitlHandler(
                enabled=True,
                result=ApprovalResult.modify('{"path":"safe.txt","content":"hello"}'),
            )
            registry = HitlToolRegistry(handler, base_dir=tmp)

            registry.execute("write_file", {"path": "unsafe.txt", "content": "hello"})

            self.assertFalse((Path(tmp) / "unsafe.txt").exists())
            self.assertEqual((Path(tmp) / "safe.txt").read_text(), "hello")


if __name__ == "__main__":
    unittest.main()
