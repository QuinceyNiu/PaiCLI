import tempfile
import unittest

from paicli.llm.glm_client import ChatResponse, Message
from paicli.memory import ContextCompressor, ConversationMemory, LongTermMemory, MemoryEntry, MemoryType


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return ChatResponse(message=Message.assistant(self.responses.pop(0)))


def conversation_entry(index: int) -> MemoryEntry:
    return MemoryEntry(
        id=f"mem_{index}",
        content=f"第 {index} 条对话内容",
        type=MemoryType.CONVERSATION,
    )


class ContextCompressorTest(unittest.TestCase):
    def test_compress_uses_map_reduce_and_reinjects_summary_with_recent_entries(self) -> None:
        llm = FakeLLMClient(["分片摘要 1", "分片摘要 2", "最终摘要"])
        compressor = ContextCompressor(llm_client=llm, retain_recent_rounds=3, chunk_size=5)
        memory = ConversationMemory(max_tokens=10_000)
        entries = [conversation_entry(index) for index in range(1, 10)]
        for entry in entries:
            memory.store(entry)

        summary = compressor.compress(memory)

        self.assertEqual(summary, "最终摘要")
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("第 1 条对话内容", llm.calls[0][0][-1].content)
        self.assertIn("第 5 条对话内容", llm.calls[0][0][-1].content)
        self.assertIn("第 6 条对话内容", llm.calls[1][0][-1].content)
        self.assertIn("分片摘要 1", llm.calls[2][0][-1].content)
        self.assertIn("分片摘要 2", llm.calls[2][0][-1].content)

        remaining = memory.entries()
        self.assertEqual(remaining[0].type, MemoryType.SUMMARY)
        self.assertTrue(remaining[0].content.startswith("[历史对话摘要] 最终摘要"))
        self.assertEqual(remaining[1:], entries[-3:])

    def test_compress_skips_reduce_when_only_one_chunk_exists(self) -> None:
        llm = FakeLLMClient(["唯一分片摘要"])
        compressor = ContextCompressor(llm_client=llm, retain_recent_rounds=3, chunk_size=5)
        memory = ConversationMemory(max_tokens=10_000)
        for index in range(1, 9):
            memory.store(conversation_entry(index))

        summary = compressor.compress(memory)

        self.assertEqual(summary, "唯一分片摘要")
        self.assertEqual(len(llm.calls), 1)

    def test_compress_includes_entries_already_evicted_for_summary(self) -> None:
        llm = FakeLLMClient(["包含淘汰消息的摘要"])
        compressor = ContextCompressor(llm_client=llm, retain_recent_rounds=1, chunk_size=5)
        memory = ConversationMemory(max_tokens=10_000)
        first = MemoryEntry(id="mem_1", content="aaaaaaaaaaaa", type=MemoryType.CONVERSATION)
        second = MemoryEntry(id="mem_2", content="bbbbbbbbbbbb", type=MemoryType.CONVERSATION)
        third = MemoryEntry(id="mem_3", content="cccccccccccc", type=MemoryType.CONVERSATION)
        memory.compressed_summaries.append(first)
        memory.store(second)
        memory.store(third)

        compressor.compress(memory)

        self.assertIn("aaaaaaaaaaaa", llm.calls[0][0][-1].content)
        self.assertEqual(memory.compressed_summaries, [])

    def test_extract_facts_stores_llm_facts_in_long_term_memory(self) -> None:
        llm = FakeLLMClient(["- 用户喜欢 Spring Boot\n- 项目使用 Maven 构建"])
        compressor = ContextCompressor(llm_client=llm)
        entries = [
            MemoryEntry(
                id="mem_1",
                content="我喜欢 Spring Boot，这个项目用 Maven 构建",
                type=MemoryType.CONVERSATION,
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            long_term_memory = LongTermMemory(storage_dir=tmp)

            facts = compressor.extract_facts(entries, long_term_memory)

            self.assertEqual(facts, ["用户喜欢 Spring Boot", "项目使用 Maven 构建"])
            self.assertEqual(
                [entry.content for entry in long_term_memory.entries()],
                ["用户喜欢 Spring Boot", "项目使用 Maven 构建"],
            )
            self.assertTrue(all(entry.type == MemoryType.FACT for entry in long_term_memory.entries()))


if __name__ == "__main__":
    unittest.main()
