from datetime import datetime, timezone
import unittest

from paicli.memory import MemoryEntry, MemoryType, estimate_tokens


class MemoryEntryTest(unittest.TestCase):
    def test_entry_keeps_memory_metadata_and_estimated_token_count(self) -> None:
        timestamp = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)

        entry = MemoryEntry(
            id="mem_1",
            content="我喜欢用 JDK 17",
            type=MemoryType.FACT,
            timestamp=timestamp,
            metadata={"project": "PaiCli"},
        )

        self.assertEqual(entry.id, "mem_1")
        self.assertEqual(entry.content, "我喜欢用 JDK 17")
        self.assertEqual(entry.type, MemoryType.FACT)
        self.assertEqual(entry.timestamp, timestamp)
        self.assertEqual(entry.metadata, {"project": "PaiCli"})
        self.assertEqual(entry.token_count, estimate_tokens("我喜欢用 JDK 17"))

    def test_estimate_tokens_counts_chinese_more_densely_than_other_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("中文"), 2)
        self.assertEqual(estimate_tokens("我喜欢用 JDK 17"), 5)


if __name__ == "__main__":
    unittest.main()
