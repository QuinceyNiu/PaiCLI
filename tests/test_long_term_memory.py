from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

from paicli.memory import LongTermMemory, MemoryEntry, MemoryQueryTokenizer, MemoryType


class LongTermMemoryTest(unittest.TestCase):
    def test_store_deduplicates_and_persists_entries_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(storage_dir=tmp)
            entry = MemoryEntry(
                id="mem_1",
                content="用户喜欢 JDK 17",
                type=MemoryType.FACT,
                metadata={"project": "PaiCli"},
            )

            memory.store(entry)
            duplicate = memory.store(
                MemoryEntry(
                    id="mem_2",
                    content="用户喜欢 JDK 17",
                    type=MemoryType.FACT,
                    metadata={"project": "PaiCli"},
                )
            )

            self.assertIsNone(duplicate)
            self.assertEqual(memory.entries(), [entry])
            self.assertTrue((Path(tmp) / "long_term_memory.json").exists())

    def test_loads_entries_from_disk_and_preserves_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timestamp = datetime(2026, 5, 16, 9, 30, tzinfo=timezone.utc)
            entry = MemoryEntry(
                id="mem_1",
                content="项目用 Maven 构建",
                type=MemoryType.FACT,
                timestamp=timestamp,
            )
            LongTermMemory(storage_dir=tmp).store(entry)

            loaded = LongTermMemory(storage_dir=tmp)

            self.assertEqual(loaded.entries(), [entry])
            self.assertEqual(loaded.entries()[0].timestamp, timestamp)

    def test_search_matches_content_and_metadata_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(storage_dir=tmp)
            jdk = MemoryEntry(
                id="mem_1",
                content="用户喜欢 JDK 17",
                type=MemoryType.FACT,
                metadata={"project": "PaiCli"},
            )
            maven = MemoryEntry(
                id="mem_2",
                content="项目用 Maven 构建",
                type=MemoryType.FACT,
                metadata={"stack": "Java Spring Boot"},
            )
            memory.store(jdk)
            memory.store(maven)

            self.assertEqual(memory.search("喜欢 JDK", limit=5), [jdk])
            self.assertEqual(memory.search("Spring Boot", limit=5), [maven])
            self.assertEqual(memory.search("不存在", limit=5), [])

    def test_tokenizer_keeps_useful_chinese_terms_when_jieba_is_unavailable(self) -> None:
        tokens = MemoryQueryTokenizer.tokenize("项目技术栈")

        self.assertIn("项目", tokens)
        self.assertIn("技术", tokens)


if __name__ == "__main__":
    unittest.main()
