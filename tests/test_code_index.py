import tempfile
import unittest
from pathlib import Path

from paicli.rag import CodeIndex, EmbeddingClient, VectorStore


class RecordingVectorStore(VectorStore):
    def __init__(self) -> None:
        super().__init__(":memory:")
        self.stored_chunks = []
        self.project_paths = []
        self.stored_relations = []
        self.cleared_project_paths = []

    def save_chunks(self, chunks, project_path: str = "") -> None:
        self.stored_chunks.extend(chunks)
        self.project_paths.append(project_path)
        super().save_chunks(chunks, project_path=project_path)

    def save_relations(self, relations, project_path: str = "") -> None:
        self.stored_relations.extend(relations)
        self.project_paths.append(project_path)
        super().save_relations(relations, project_path=project_path)

    def clear(self, project_path: str | None = None) -> None:
        self.cleared_project_paths.append(project_path)
        super().clear(project_path=project_path)


class CodeIndexTest(unittest.TestCase):
    def test_index_walks_supported_files_and_skips_build_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("class App:\n    def run(self):\n        pass\n", encoding="utf-8")
            (root / "README.md").write_text("PaiCli docs", encoding="utf-8")
            (root / "target").mkdir()
            (root / "target" / "Ignored.py").write_text("class Ignored:\n    pass\n", encoding="utf-8")
            (root / "image.png").write_text("not source", encoding="utf-8")
            store = RecordingVectorStore()

            indexed = CodeIndex(
                vector_store=store,
                embedding_client=EmbeddingClient(provider="local"),
            ).index(root)

        paths = {chunk.file_path for chunk in store.stored_chunks}
        self.assertEqual(indexed, len(store.stored_chunks))
        self.assertIn("src/app.py", paths)
        self.assertIn("README.md", paths)
        self.assertNotIn("target/Ignored.py", paths)
        self.assertNotIn("image.png", paths)
        self.assertEqual(set(store.project_paths), {str(root.resolve())})
        self.assertEqual(store.cleared_project_paths, [str(root.resolve())])

    def test_index_persists_code_relations_for_supported_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "agent.py").write_text(
                "class Agent(BaseAgent):\n"
                "    def run(self):\n"
                "        self.chat()\n",
                encoding="utf-8",
            )
            store = RecordingVectorStore()

            CodeIndex(
                vector_store=store,
                embedding_client=EmbeddingClient(provider="local"),
            ).index(root)

        rendered = {
            (relation.source, relation.target, relation.relation_type, relation.file_path)
            for relation in store.stored_relations
        }
        self.assertIn(("Agent", "BaseAgent", "extends", "src/agent.py"), rendered)
        self.assertIn(("Agent", "Agent.run", "contains", "src/agent.py"), rendered)
        self.assertIn(("Agent.run", "chat", "calls", "src/agent.py"), rendered)

    def test_index_with_stats_reports_progress_every_ten_files_and_continues_after_failure(self) -> None:
        class FailingChunker:
            def __init__(self) -> None:
                self.calls = 0

            def chunk_file(self, path):
                self.calls += 1
                if path.name == "file_11.py":
                    raise ValueError("broken")
                return [CodeChunk.file_chunk(path.name, f"{path.name}#1", "content", 1, 1)]

        from paicli.rag import CodeChunk

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(1, 13):
                (root / f"file_{index}.py").write_text("print('x')\n", encoding="utf-8")
            progress = []
            warnings = []
            store = RecordingVectorStore()

            stats = CodeIndex(
                vector_store=store,
                chunker=FailingChunker(),
                embedding_client=EmbeddingClient(provider="local"),
                on_progress=progress.append,
                on_warning=warnings.append,
            ).index_with_stats(root)

        self.assertEqual(stats.file_count, 12)
        self.assertEqual(stats.chunk_count, 11)
        self.assertEqual(stats.failed_files, 1)
        self.assertEqual([item.processed_files for item in progress], [10, 12])
        self.assertIn("file_11.py", warnings[0])


if __name__ == "__main__":
    unittest.main()
