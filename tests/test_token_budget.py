import unittest

from paicli.memory import ConversationMemory, MemoryEntry, MemoryType, TokenBudget


class TokenBudgetTest(unittest.TestCase):
    def test_available_conversation_budget_reserves_system_tools_and_response_tokens(self) -> None:
        budget = TokenBudget(
            context_window=200_000,
            reserved_for_system=500,
            reserved_for_tools=800,
            reserved_for_response=2_000,
        )

        self.assertEqual(budget.get_available_for_conversation(), 196_700)

    def test_needs_compression_when_conversation_exceeds_eighty_percent_of_available_budget(self) -> None:
        budget = TokenBudget(
            context_window=100,
            reserved_for_system=10,
            reserved_for_tools=10,
            reserved_for_response=10,
        )
        memory = ConversationMemory(max_tokens=1_000)
        memory.store(MemoryEntry(id="mem_1", content="a" * 224, type=MemoryType.CONVERSATION))

        self.assertFalse(budget.needs_compression(memory))

        memory.store(MemoryEntry(id="mem_2", content="bbbb", type=MemoryType.CONVERSATION))

        self.assertTrue(budget.needs_compression(memory))

    def test_records_llm_usage_and_formats_report(self) -> None:
        budget = TokenBudget()

        budget.record_usage(input_tokens=312, output_tokens=156)
        budget.record_usage(input_tokens=100, output_tokens=50)

        self.assertIn("Token 统计: 调用 2 次 | 总输入: 412 | 总输出: 206", budget.get_usage_report())
        self.assertIn("平均输入: 206", budget.get_usage_report())


if __name__ == "__main__":
    unittest.main()
