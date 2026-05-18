from datetime import datetime, timezone
import unittest

from paicli.memory import ConversationMemory, MemoryEntry, MemoryType


class ConversationMemoryTest(unittest.TestCase):
    def test_store_evicts_oldest_entries_into_compressed_summaries(self) -> None:
        memory = ConversationMemory(max_tokens=6)
        first = MemoryEntry(
            id="mem_1",
            content="abcdefghijkl",
            type=MemoryType.CONVERSATION,
            timestamp=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        )
        second = MemoryEntry(
            id="mem_2",
            content="mnopqrst",
            type=MemoryType.CONVERSATION,
            timestamp=datetime(2026, 5, 16, 10, 1, tzinfo=timezone.utc),
        )
        third = MemoryEntry(
            id="mem_3",
            content="uvwxyzab",
            type=MemoryType.CONVERSATION,
            timestamp=datetime(2026, 5, 16, 10, 2, tzinfo=timezone.utc),
        )

        memory.store(first)
        memory.store(second)
        memory.store(third)

        self.assertEqual(memory.entries(), [second, third])
        self.assertEqual(memory.compressed_summaries, [first])
        self.assertEqual(memory.current_tokens, second.token_count + third.token_count)

    def test_usage_ratio_reports_current_token_pressure(self) -> None:
        memory = ConversationMemory(max_tokens=10)

        memory.store(MemoryEntry(id="mem_1", content="abcdefgh", type=MemoryType.CONVERSATION))

        self.assertEqual(memory.get_usage_ratio(), 0.2)


if __name__ == "__main__":
    unittest.main()
