import tempfile
import unittest
from pathlib import Path

from paicli.rag import CodeChunk, CodeChunker, ChunkType


class CodeChunkTest(unittest.TestCase):
    def test_to_embedding_text_prefixes_chunk_type_and_name(self) -> None:
        chunk = CodeChunk.method_chunk(
            file_path="paicli/agent/agent.py",
            name="Agent.run",
            content="def run(self, user_input):\n    return user_input",
            start_line=10,
            end_line=11,
        )

        self.assertEqual(
            chunk.to_embedding_text(),
            "[method:Agent.run] def run(self, user_input):\n    return user_input",
        )


class CodeChunkerTest(unittest.TestCase):
    def test_java_file_is_split_into_class_header_and_method_chunks(self) -> None:
        source = "\n".join(
            [
                "package demo;",
                "",
                "public class LoginService {",
                "    private final String salt;",
                "",
                "    public LoginService(String salt) {",
                "        this.salt = salt;",
                "    }",
                "",
                "    public boolean authenticate(String user) {",
                "        if (user == null) {",
                "            return false;",
                "        }",
                "        return true;",
                "    }",
                "}",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LoginService.java"
            path.write_text(source, encoding="utf-8")

            chunks = CodeChunker(project_root=tmp).chunk_file(path)

        self.assertEqual([chunk.type for chunk in chunks], [ChunkType.CLASS, ChunkType.METHOD, ChunkType.METHOD])
        self.assertEqual(chunks[0].name, "LoginService")
        self.assertEqual(chunks[0].start_line, 3)
        self.assertEqual(chunks[0].end_line, 7)
        self.assertIn("public class LoginService", chunks[0].content)
        self.assertNotIn("return true", chunks[0].content)
        self.assertEqual(chunks[1].name, "LoginService.LoginService(String salt)")
        self.assertEqual(chunks[1].start_line, 6)
        self.assertEqual(chunks[1].end_line, 8)
        self.assertEqual(chunks[2].name, "LoginService.authenticate(String user)")
        self.assertEqual(chunks[2].start_line, 10)
        self.assertEqual(chunks[2].end_line, 15)
        self.assertIn("return true", chunks[2].content)

    def test_java_parse_failure_falls_back_to_text_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Broken.java"
            path.write_text("public class Broken {\n    public void run( {\n", encoding="utf-8")

            chunks = CodeChunker(project_root=tmp).chunk_file(path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].type, ChunkType.FILE)
        self.assertEqual(chunks[0].name, "Broken.java#1")

    def test_python_file_is_split_into_class_header_and_method_chunks(self) -> None:
        source = "\n".join(
            [
                "class LoginService:",
                "    token = 'x'",
                "",
                "    def authenticate(self, user):",
                "        if not user:",
                "            return False",
                "        return True",
                "",
                "def helper():",
                "    return 'ok'",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.py"
            path.write_text(source, encoding="utf-8")

            chunks = CodeChunker(project_root=tmp).chunk_file(path)

        self.assertEqual([chunk.type for chunk in chunks], [ChunkType.CLASS, ChunkType.METHOD])
        self.assertEqual(chunks[0].name, "LoginService")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 5)
        self.assertIn("class LoginService:", chunks[0].content)
        self.assertIn("token = 'x'", chunks[0].content)
        self.assertNotIn("return True", chunks[0].content)
        self.assertEqual(chunks[1].name, "LoginService.authenticate")
        self.assertEqual(chunks[1].start_line, 4)
        self.assertEqual(chunks[1].end_line, 7)
        self.assertIn("return True", chunks[1].content)

    def test_syntax_error_falls_back_to_text_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text("class Broken(:\n    pass\n", encoding="utf-8")

            chunks = CodeChunker(project_root=tmp).chunk_file(path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].type, ChunkType.FILE)
        self.assertEqual(chunks[0].name, "broken.py#1")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 2)

    def test_non_source_text_is_split_into_two_thousand_character_chunks_with_lines(self) -> None:
        content = ("a\n" * 1200) + ("b\n" * 1200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README.md"
            path.write_text(content, encoding="utf-8")

            chunks = CodeChunker(project_root=tmp, max_text_chunk_chars=2000).chunk_file(path)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.type == ChunkType.FILE for chunk in chunks))
        self.assertEqual(chunks[0].start_line, 1)
        self.assertLess(chunks[0].end_line, chunks[-1].end_line)
        self.assertLessEqual(max(len(chunk.content) for chunk in chunks), 2000)


if __name__ == "__main__":
    unittest.main()
