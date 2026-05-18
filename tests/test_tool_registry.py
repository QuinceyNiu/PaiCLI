import json
import tempfile
import unittest
from pathlib import Path

from paicli.rag import CodeChunk, SearchResult
from paicli.tool.tool_registry import ToolRegistry, create_parameters


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def hybrid_search(self, query, top_k=5, project_path="", max_per_file=2):
        self.calls.append((query, top_k, project_path, max_per_file))
        return [
            SearchResult(
                CodeChunk.method_chunk(
                    "agent.py",
                    "Agent.run",
                    "def run(self):\n    return 'ok'",
                    10,
                    11,
                ),
                1.0,
            )
        ]


class ToolRegistryTest(unittest.TestCase):
    def test_create_parameters_builds_json_schema(self) -> None:
        schema = create_parameters(("path", "string", "文件路径", True))

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["path"]["type"], "string")
        self.assertEqual(schema["required"], ["path"])

    def test_file_tools_read_write_and_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(base_dir=tmp)

            write_result = registry.execute(
                "write_file",
                {"path": "notes/hello.txt", "content": "hello pai"},
            )
            read_result = registry.execute("read_file", {"path": "notes/hello.txt"})
            list_result = registry.execute("list_dir", {"path": "notes"})

            self.assertIn("文件已写入", write_result)
            self.assertIn("hello pai", read_result)
            self.assertIn("hello.txt", list_result)

    def test_execute_command_returns_exit_code_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(base_dir=tmp)

            result = registry.execute("execute_command", {"command": "printf pai"})

            self.assertIn("exit code: 0", result)
            self.assertIn("pai", result)

    def test_create_project_creates_directories_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(base_dir=tmp)
            files = {"README.md": "# Demo", "src/main.py": "print('hi')\n"}

            result = registry.execute(
                "create_project",
                {"path": "demo", "files": json.dumps(files)},
            )

            self.assertIn("项目已创建", result)
            self.assertEqual((Path(tmp) / "demo" / "README.md").read_text(), "# Demo")
            self.assertTrue((Path(tmp) / "demo" / "src" / "main.py").exists())

    def test_tools_for_llm_omits_executors(self) -> None:
        registry = ToolRegistry()

        tools = registry.tools_for_llm()

        self.assertTrue(any(tool.name == "read_file" for tool in tools))
        self.assertTrue(all(hasattr(tool, "parameters") for tool in tools))

    def test_search_code_tool_uses_retriever_and_formats_results(self) -> None:
        retriever = FakeRetriever()
        registry = ToolRegistry(
            base_dir="/repo",
            code_retriever_factory=lambda: retriever,
        )
        registry.set_project_path("/indexed")

        result = registry.execute("search_code", {"query": "Agent run", "top_k": "3"})

        self.assertEqual(retriever.calls, [("Agent run", 3, "/indexed", 2)])
        self.assertIn("查询: Agent run", result)
        self.assertIn("Agent.run", result)
        self.assertIn("agent.py:10-11", result)

    def test_search_code_tool_defaults_top_k_and_reports_missing_query(self) -> None:
        registry = ToolRegistry(code_retriever_factory=FakeRetriever)

        self.assertIn("query 不能为空", registry.execute("search_code", {"query": ""}))


if __name__ == "__main__":
    unittest.main()
