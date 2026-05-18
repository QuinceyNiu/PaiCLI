import unittest
import tempfile

from paicli.memory import (
    ContextCompressor,
    ConversationMemory,
    LongTermMemory,
    MemoryManager,
    MemoryRetriever,
    MemoryType,
    TokenBudget,
)


class RecordingCompressor(ContextCompressor):
    def __init__(self) -> None:
        self.calls = []

    def compress(self, memory):
        self.calls.append(memory)
        return "压缩摘要"


class MemoryManagerTest(unittest.TestCase):
    def test_manager_exposes_store_message_and_get_memory_facade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))

            entry = manager.store_message("用户喜欢 JDK 17", MemoryType.FACT, {"project": "PaiCli"})
            memories = manager.get_memory("JDK")

            self.assertEqual(entry.type, MemoryType.FACT)
            self.assertEqual(entry.metadata, {"project": "PaiCli"})
            self.assertEqual(memories, [entry])

    def test_manager_owns_all_memory_components(self) -> None:
        manager = MemoryManager()

        self.assertIsInstance(manager.conversation_memory, ConversationMemory)
        self.assertIsInstance(manager.long_term_memory, LongTermMemory)
        self.assertIsInstance(manager.context_compressor, ContextCompressor)
        self.assertIsInstance(manager.token_budget, TokenBudget)
        self.assertIsInstance(manager.memory_retriever, MemoryRetriever)
        self.assertEqual(
            manager.conversation_memory.max_tokens,
            manager.token_budget.get_available_for_conversation(),
        )

    def test_store_message_triggers_compression_when_conversation_memory_is_nearly_full(self) -> None:
        compressor = RecordingCompressor()
        manager = MemoryManager(
            conversation_memory=ConversationMemory(max_tokens=1_000),
            context_compressor=compressor,
            token_budget=TokenBudget(
                context_window=20,
                reserved_for_system=0,
                reserved_for_tools=0,
                reserved_for_response=0,
            ),
        )

        entry = manager.store_message("a" * 68)

        self.assertEqual(compressor.calls, [manager.conversation_memory])
        self.assertEqual(manager.conversation_memory.entries(), [entry])

    def test_convenience_methods_store_user_assistant_tool_and_manual_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))

            user = manager.add_user_message("我喜欢 JDK 17")
            assistant = manager.add_assistant_message("好的，我会记住")
            tool = manager.add_tool_result("read_file", "文件内容")
            fact = manager.save_fact("用户喜欢 JDK 17")

            self.assertEqual(user.type, MemoryType.CONVERSATION)
            self.assertEqual(assistant.type, MemoryType.CONVERSATION)
            self.assertEqual(tool.type, MemoryType.TOOL_RESULT)
            self.assertEqual(tool.metadata, {"tool": "read_file"})
            self.assertEqual(fact.type, MemoryType.FACT)
            self.assertEqual(manager.long_term_memory.entries(), [fact])

    def test_extract_and_save_facts_uses_short_term_memory_then_can_clear_it(self) -> None:
        class FakeCompressor(ContextCompressor):
            def __init__(self) -> None:
                self.entries = []

            def extract_facts(self, entries, long_term_memory):
                self.entries = list(entries)
                long_term_memory.store(
                    manager.save_fact("项目使用 Maven 构建", metadata={"source": "clear"})
                )
                return ["项目使用 Maven 构建"]

        with tempfile.TemporaryDirectory() as tmp:
            compressor = FakeCompressor()
            manager = MemoryManager(
                long_term_memory=LongTermMemory(storage_dir=tmp),
                context_compressor=compressor,
            )
            manager.add_user_message("项目使用 Maven 构建")

            facts = manager.extract_and_save_facts()
            manager.clear_short_term()

            self.assertEqual(facts, ["项目使用 Maven 构建"])
            self.assertEqual([entry.content for entry in compressor.entries], ["项目使用 Maven 构建"])
            self.assertEqual(manager.conversation_memory.entries(), [])

    def test_build_context_for_query_respects_max_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))
            manager.save_fact("用户喜欢 JDK 17")
            manager.save_fact("用户喜欢 JDK 21")

            context = manager.build_context_for_query("JDK", max_tokens=10)

            self.assertIn("用户喜欢 JDK", context)
            self.assertLessEqual(sum(line.count("JDK") for line in context.splitlines()), 1)

    def test_get_system_status_combines_memory_and_token_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))
            manager.add_user_message("我喜欢 JDK 17")
            manager.save_fact("用户喜欢 JDK 17")
            manager.record_token_usage(12, 5)

            status = manager.get_system_status()

            self.assertIn("短期记忆: 1条", status)
            self.assertIn("长期记忆: 1条", status)
            self.assertIn("Token 统计: 调用 1 次 | 总输入: 12 | 总输出: 5", status)


if __name__ == "__main__":
    unittest.main()
