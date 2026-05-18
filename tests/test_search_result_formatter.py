import unittest

from paicli.rag import CodeChunk, SearchResult, SearchResultFormatter


class SearchResultFormatterTest(unittest.TestCase):
    def test_format_for_tool_is_compact_and_includes_summary_and_code(self) -> None:
        results = [
            SearchResult(
                CodeChunk.method_chunk(
                    "paicli/agent/agent.py",
                    "Agent.run",
                    "def run(self, user_input):\n    return user_input",
                    40,
                    41,
                ),
                1.42,
            ),
            SearchResult(
                CodeChunk.class_chunk(
                    "paicli/agent/agent.py",
                    "Agent",
                    "class Agent:",
                    20,
                    20,
                ),
                1.10,
            ),
        ]

        formatted = SearchResultFormatter.format_for_tool("Agent 的 run 方法", results)

        self.assertIn("查询: Agent 的 run 方法", formatted)
        self.assertIn("最相关入口: Agent.run", formatted)
        self.assertIn("主要文件: paicli/agent/agent.py", formatted)
        self.assertIn("1. [method] Agent.run", formatted)
        self.assertIn("paicli/agent/agent.py:40-41", formatted)
        self.assertIn("def run(self, user_input):", formatted)

    def test_format_for_cli_uses_readable_tree_output(self) -> None:
        formatted = SearchResultFormatter.format_for_cli(
            "compress",
            [
                SearchResult(
                    CodeChunk.method_chunk("memory.py", "MemoryManager.compress_if_needed", "body", 10, 12),
                    1.23,
                )
            ],
        )

        self.assertIn("🔎 检索: compress", formatted)
        self.assertIn("找到 1 个相关代码块", formatted)
        self.assertIn("memory.py:10-12", formatted)
        self.assertIn("MemoryManager.compress_if_needed", formatted)


if __name__ == "__main__":
    unittest.main()
