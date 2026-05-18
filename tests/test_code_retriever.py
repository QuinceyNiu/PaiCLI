import unittest

from paicli.rag import CodeChunk, CodeRetriever, EmbeddingClient, RagQueryTokenizer, VectorStore


class StaticEmbeddingClient(EmbeddingClient):
    def __init__(self, vector):
        self.vector = vector

    def embed(self, text: str):
        return self.vector


class RagQueryTokenizerTest(unittest.TestCase):
    def test_tokenize_keeps_ascii_identifiers_and_filters_low_value_words(self) -> None:
        tokens = RagQueryTokenizer.tokenize("Agent 的 run 方法怎么实现一下？MemoryManager.compressIfNeeded")

        self.assertIn("Agent", tokens)
        self.assertIn("run", tokens)
        self.assertIn("MemoryManager", tokens)
        self.assertIn("compressIfNeeded", tokens)
        self.assertNotIn("怎么", tokens)
        self.assertNotIn("一下", tokens)
        self.assertNotIn("的", tokens)

    def test_tokenize_expands_common_chinese_code_terms_to_identifiers(self) -> None:
        tokens = RagQueryTokenizer.tokenize("MemoryManager 是怎么压缩上下文的")

        self.assertIn("MemoryManager", tokens)
        self.assertIn("compress", tokens)
        self.assertIn("context", tokens)


class CodeRetrieverTest(unittest.TestCase):
    def test_hybrid_search_boosts_exact_method_identifier_over_semantic_only_match(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.method_chunk(
                    "agent.py",
                    "Agent.run",
                    "def run(self, user_input): react loop",
                    10,
                    20,
                ).with_embedding([0.7, 0.7]),
                CodeChunk.method_chunk(
                    "planner.py",
                    "Planner.execute",
                    "agent context and running task",
                    1,
                    8,
                ).with_embedding([1.0, 0.0]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(
            vector_store=store,
            embedding_client=StaticEmbeddingClient([1.0, 0.0]),
        )

        results = retriever.hybrid_search("Agent 的 run 方法", top_k=2, project_path="/repo")

        self.assertEqual(results[0].chunk.name, "Agent.run")
        self.assertGreater(results[0].similarity, results[1].similarity)

    def test_hybrid_search_prefers_real_memory_compression_over_test_fakes(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.method_chunk(
                    "paicli/memory/manager.py",
                    "MemoryManager.compress_if_needed",
                    "def compress_if_needed(self):\n    if self.token_budget.needs_compression(self.conversation_memory):\n        self.context_compressor.compress(self.conversation_memory)",
                    390,
                    394,
                ).with_embedding([0.7, 0.7]),
                CodeChunk.method_chunk(
                    "paicli/memory/manager.py",
                    "ContextCompressor.compress",
                    "def compress(self, memory):\n    final_summary = self._reduce_phase(chunk_summaries)\n    memory.store(summary)",
                    120,
                    140,
                ).with_embedding([0.7, 0.7]),
                CodeChunk.method_chunk(
                    "tests/test_memory_manager.py",
                    "MemoryManagerTest.test_convenience_methods_store_user_assistant_tool_and_manual_facts",
                    "manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))",
                    67,
                    81,
                ).with_embedding([1.0, 0.0]),
                CodeChunk.method_chunk(
                    "tests/test_code_retriever.py",
                    "CodeRetrieverTest.test_hybrid_search_prefers_real_memory_compression_over_test_fakes",
                    "tokens = RagQueryTokenizer.tokenize('MemoryManager 是怎么压缩上下文的')\nself.assertIn('compress', tokens)\nself.assertIn('ContextCompressor.compress', names)",
                    66,
                    99,
                ).with_embedding([1.0, 0.0]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(store, StaticEmbeddingClient([1.0, 0.0]))

        results = retriever.hybrid_search("MemoryManager 是怎么压缩上下文的", top_k=3, project_path="/repo")

        self.assertEqual(results[0].chunk.name, "MemoryManager.compress_if_needed")
        self.assertIn("ContextCompressor.compress", [result.chunk.name for result in results[:2]])

    def test_hybrid_search_boosts_explicit_identifier_over_context_only_matches(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.class_chunk(
                    "paicli/memory/manager.py",
                    "ContextCompressor",
                    "class ContextCompressor:\n    MAP_PROMPT = '压缩上下文摘要'",
                    147,
                    151,
                ).with_embedding([1.0, 0.0]),
                CodeChunk.method_chunk(
                    "paicli/agent/agent_orchestrator.py",
                    "AgentOrchestrator.build_step_context",
                    "def build_step_context(self):\n    lines = ['总任务上下文']",
                    226,
                    238,
                ).with_embedding([1.0, 0.0]),
                CodeChunk.class_chunk(
                    "paicli/rag/retriever.py",
                    "RagQueryTokenizer",
                    'SYNONYMS = {"压缩": ("compress",), "上下文": ("context",)}',
                    13,
                    17,
                ).with_embedding([1.0, 0.0]),
                CodeChunk.method_chunk(
                    "paicli/memory/manager.py",
                    "MemoryManager.compress_if_needed",
                    "def compress_if_needed(self):\n    self.context_compressor.compress(self.conversation_memory)",
                    500,
                    503,
                ).with_embedding([0.4, 0.4]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(store, StaticEmbeddingClient([1.0, 0.0]))

        results = retriever.hybrid_search("MemoryManager 是怎么压缩上下文的", top_k=4, project_path="/repo")

        self.assertEqual(results[0].chunk.name, "MemoryManager.compress_if_needed")

    def test_hybrid_search_does_not_penalize_tests_when_query_asks_for_tests(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.method_chunk(
                    "paicli/memory/manager.py",
                    "MemoryManager.compress_if_needed",
                    "def compress_if_needed(self): pass",
                    390,
                    394,
                ).with_embedding([0.5, 0.5]),
                CodeChunk.method_chunk(
                    "tests/test_memory_manager.py",
                    "MemoryManagerTest.test_compress_if_needed",
                    "def test_compress_if_needed(self): pass",
                    10,
                    12,
                ).with_embedding([1.0, 0.0]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(store, StaticEmbeddingClient([1.0, 0.0]))

        results = retriever.hybrid_search("MemoryManager 压缩上下文的测试", top_k=2, project_path="/repo")

        self.assertEqual(results[0].chunk.name, "MemoryManagerTest.test_compress_if_needed")

    def test_hybrid_search_adds_type_boost_and_double_hit_bonus_once(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.file_chunk("agent.py", "agent.py#1", "Agent overview", 1, 5).with_embedding([1, 0]),
                CodeChunk.class_chunk("agent.py", "Agent", "class Agent", 1, 5).with_embedding([1, 0]),
                CodeChunk.method_chunk("agent.py", "Agent.run", "def run(self): pass", 6, 10).with_embedding([1, 0]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(store, StaticEmbeddingClient([1, 0]))

        results = retriever.hybrid_search("Agent run", top_k=3, project_path="/repo", max_per_file=3)

        self.assertEqual([result.chunk.type.value for result in results], ["method", "class", "file"])
        self.assertAlmostEqual(results[0].similarity, 2.85)
        self.assertAlmostEqual(results[1].similarity, 2.5)
        self.assertAlmostEqual(results[2].similarity, 2.4)

    def test_hybrid_search_limits_results_per_file_for_diversity(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.method_chunk("big.py", "Big.one", "target", 1, 2).with_embedding([1, 0]),
                CodeChunk.method_chunk("big.py", "Big.two", "target", 3, 4).with_embedding([0.99, 0]),
                CodeChunk.method_chunk("big.py", "Big.three", "target", 5, 6).with_embedding([0.98, 0]),
                CodeChunk.method_chunk("other.py", "Other.one", "target", 1, 2).with_embedding([0.5, 0]),
            ],
            project_path="/repo",
        )
        retriever = CodeRetriever(store, StaticEmbeddingClient([1, 0]))

        results = retriever.hybrid_search("target", top_k=3, project_path="/repo", max_per_file=2)

        self.assertEqual([result.chunk.name for result in results], ["Big.one", "Big.two", "Other.one"])


if __name__ == "__main__":
    unittest.main()
