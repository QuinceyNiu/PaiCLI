from datetime import datetime, timedelta, timezone
import tempfile
import unittest

from paicli.memory import (
    ConversationMemory,
    LongTermMemory,
    MemoryEntry,
    MemoryManager,
    MemoryRetriever,
    MemoryType,
)


class MemoryRetrieverTest(unittest.TestCase):
    def test_retrieve_scores_short_and_long_term_memory_by_relevance_and_source_weight(self) -> None:
        now = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
        retriever = MemoryRetriever(now_func=lambda: now)
        short_term = ConversationMemory()
        long_term = LongTermMemory(storage_dir=tempfile.mkdtemp())
        short = MemoryEntry(
            id="short_1",
            content="用户喜欢 JDK 17",
            type=MemoryType.CONVERSATION,
            timestamp=now,
        )
        long = MemoryEntry(
            id="long_1",
            content="用户喜欢 JDK 17",
            type=MemoryType.FACT,
            timestamp=now,
        )
        short_term.store(short)
        long_term.store(long)

        results = retriever.retrieve("JDK 17", short_term, long_term, limit=2)

        self.assertEqual(results, [long, short])

    def test_retrieve_applies_time_decay_to_older_entries(self) -> None:
        now = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
        retriever = MemoryRetriever(now_func=lambda: now)
        short_term = ConversationMemory()
        long_term = LongTermMemory(storage_dir=tempfile.mkdtemp())
        old = MemoryEntry(
            id="old",
            content="项目使用 Maven 构建",
            type=MemoryType.CONVERSATION,
            timestamp=now - timedelta(days=3),
        )
        recent = MemoryEntry(
            id="recent",
            content="项目使用 Maven 构建",
            type=MemoryType.CONVERSATION,
            timestamp=now - timedelta(hours=1),
        )
        short_term.store(old)
        short_term.store(recent)

        results = retriever.retrieve("Maven 构建", short_term, long_term, limit=2)

        self.assertEqual(results, [recent, old])

    def test_build_context_for_query_formats_limited_relevant_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))
            manager.store_message("用户喜欢 JDK 17", MemoryType.FACT)
            manager.store_message("无关闲聊内容", MemoryType.CONVERSATION)

            context = manager.build_context_for_query("JDK", limit=3)

            self.assertIn("[相关记忆]", context)
            self.assertIn("- FACT: 用户喜欢 JDK 17", context)
            self.assertNotIn("无关闲聊内容", context)


if __name__ == "__main__":
    unittest.main()
