import tempfile
import unittest
from pathlib import Path

from paicli.rag import CodeAnalyzer


class CodeAnalyzerTest(unittest.TestCase):
    def test_java_analyzer_extracts_class_method_import_and_call_relations(self) -> None:
        source = "\n".join(
            [
                "package demo;",
                "import java.util.List;",
                "import com.acme.BaseAgent;",
                "import com.acme.Tool;",
                "",
                "public class Agent extends BaseAgent implements Tool {",
                "    public String run(String input) {",
                "        chat(input);",
                "        return executeTool(input);",
                "    }",
                "}",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Agent.java"
            path.write_text(source, encoding="utf-8")

            relations = CodeAnalyzer(project_root=tmp).analyze_file(path)

        rendered = {
            (relation.source, relation.target, relation.relation_type, relation.file_path)
            for relation in relations
        }
        self.assertIn(("Agent.java", "com.acme.BaseAgent", "imports", "Agent.java"), rendered)
        self.assertIn(("Agent.java", "com.acme.Tool", "imports", "Agent.java"), rendered)
        self.assertNotIn(("Agent.java", "java.util.List", "imports", "Agent.java"), rendered)
        self.assertIn(("Agent", "BaseAgent", "extends", "Agent.java"), rendered)
        self.assertIn(("Agent", "Tool", "implements", "Agent.java"), rendered)
        self.assertIn(("Agent", "Agent.run", "contains", "Agent.java"), rendered)
        self.assertIn(("Agent.run", "chat", "calls", "Agent.java"), rendered)
        self.assertIn(("Agent.run", "executeTool", "calls", "Agent.java"), rendered)

    def test_python_analyzer_extracts_imports_extends_contains_and_calls(self) -> None:
        source = "\n".join(
            [
                "import os",
                "from paicli.llm.glm_client import GLMClient",
                "",
                "class Agent(BaseAgent):",
                "    def run(self, text):",
                "        self.chat(text)",
                "        execute_tool(text)",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.py"
            path.write_text(source, encoding="utf-8")

            relations = CodeAnalyzer(project_root=tmp).analyze_file(path)

        rendered = {
            (relation.source, relation.target, relation.relation_type, relation.file_path)
            for relation in relations
        }
        self.assertIn(("agent.py", "paicli.llm.glm_client.GLMClient", "imports", "agent.py"), rendered)
        self.assertNotIn(("agent.py", "os", "imports", "agent.py"), rendered)
        self.assertIn(("Agent", "BaseAgent", "extends", "agent.py"), rendered)
        self.assertIn(("Agent", "Agent.run", "contains", "agent.py"), rendered)
        self.assertIn(("Agent.run", "chat", "calls", "agent.py"), rendered)
        self.assertIn(("Agent.run", "execute_tool", "calls", "agent.py"), rendered)


if __name__ == "__main__":
    unittest.main()
