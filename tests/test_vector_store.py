import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paicli.rag import ChunkType, CodeChunk, CodeRelation, VectorStore


class VectorStoreTest(unittest.TestCase):
    def test_search_returns_top_k_by_cosine_similarity_within_project(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.method_chunk("a.py", "A.close", "close file", 1, 3).with_embedding([1, 0]),
                CodeChunk.method_chunk("b.py", "B.open", "open file", 1, 3).with_embedding([0, 1]),
                CodeChunk.method_chunk("c.py", "C.mixed", "mixed", 1, 3).with_embedding([0.5, 0.5]),
            ],
            project_path="/repo",
        )
        store.save_chunks(
            [
                CodeChunk.method_chunk("foreign.py", "Foreign.close", "foreign", 1, 3).with_embedding([1, 0]),
                CodeChunk.method_chunk("other.py", "Other.close", "other", 1, 3).with_embedding([1, 0]),
            ],
            project_path="/other",
        )

        results = store.search([1, 0], top_k=2, project_path="/repo")

        self.assertEqual([result.chunk.name for result in results], ["A.close", "C.mixed"])
        self.assertEqual(results[0].similarity, 1.0)
        self.assertGreater(results[1].similarity, 0.7)

    def test_keyword_search_uses_like_on_name_and_content(self) -> None:
        store = VectorStore(":memory:")
        store.save_chunks(
            [
                CodeChunk.class_chunk("login.py", "LoginService", "class LoginService", 1, 5),
                CodeChunk.method_chunk("memory.py", "MemoryManager.compress_if_needed", "compress context", 10, 15),
            ],
            project_path="/repo",
        )

        results = store.keyword_search("compress", top_k=5, project_path="/repo")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.name, "MemoryManager.compress_if_needed")
        self.assertGreater(results[0].similarity, 0)

    def test_relations_are_saved_and_queried_by_source_or_target(self) -> None:
        store = VectorStore(":memory:")
        store.save_relations(
            [
                CodeRelation(
                    source="LoginService",
                    target="UserRepository",
                    relation_type="imports",
                    file_path="login.py",
                ),
                CodeRelation(
                    source="LoginService.authenticate",
                    target="PasswordHasher.hash",
                    relation_type="calls",
                    file_path="login.py",
                ),
            ],
            project_path="/repo",
        )

        outgoing = store.find_relations(project_path="/repo", source="LoginService.authenticate")
        incoming = store.find_relations(project_path="/repo", target="UserRepository")

        self.assertEqual([relation.relation_type for relation in outgoing], ["calls"])
        self.assertEqual([relation.relation_type for relation in incoming], ["imports"])

    def test_batch_insert_rolls_back_when_one_chunk_is_invalid(self) -> None:
        store = VectorStore(":memory:")
        valid = CodeChunk.file_chunk("ok.py", "ok.py#1", "ok", 1, 1)
        invalid = CodeChunk(
            file_path="bad.py",
            type=ChunkType.FILE,
            name="bad.py#1",
            content="bad",
            start_line=None,  # type: ignore[arg-type]
            end_line=1,
        )

        with self.assertRaises(sqlite3.IntegrityError):
            store.save_chunks([valid, invalid], project_path="/repo")

        self.assertEqual(store.all_chunks(project_path="/repo"), [])

    def test_default_database_path_uses_paicli_rag_dir_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PAICLI_RAG_DIR": tmp}, clear=False):
                store = VectorStore.default()

            self.assertEqual(Path(store.db_path), Path(tmp) / "codebase.db")


if __name__ == "__main__":
    unittest.main()
