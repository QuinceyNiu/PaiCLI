"""SQLite-backed vector storage for code chunks."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

from paicli.rag.models import ChunkType, CodeChunk, CodeRelation, SearchResult


class VectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._initialize()

    @classmethod
    def default(cls) -> "VectorStore":
        rag_dir = os.getenv("PAICLI_RAG_DIR")
        root = Path(rag_dir).expanduser() if rag_dir else Path.home() / ".paicli" / "rag"
        return cls(root / "codebase.db")

    def save_chunks(self, chunks: list[CodeChunk], project_path: str = "") -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO code_chunks
                    (project_path, file_path, chunk_type, name, content, start_line, end_line, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        project_path,
                        chunk.file_path,
                        chunk.type.value,
                        chunk.name,
                        chunk.content,
                        chunk.start_line,
                        chunk.end_line,
                        json.dumps(chunk.embedding),
                    )
                    for chunk in chunks
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        project_path: str = "",
    ) -> list[SearchResult]:
        candidates = [
            SearchResult(
                chunk=chunk,
                similarity=self._cosine_similarity(query_embedding, chunk.embedding),
            )
            for chunk in self.all_chunks(project_path=project_path)
            if chunk.embedding
        ]
        candidates.sort(key=lambda result: result.similarity, reverse=True)
        return candidates[:top_k]

    def keyword_search(
        self,
        keyword: str,
        top_k: int,
        project_path: str = "",
    ) -> list[SearchResult]:
        pattern = f"%{keyword}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT file_path, chunk_type, name, content, start_line, end_line, embedding_json
                FROM code_chunks
                WHERE project_path = ?
                  AND (name LIKE ? OR content LIKE ? OR file_path LIKE ?)
                ORDER BY id
                LIMIT ?
                """,
                (project_path, pattern, pattern, pattern, top_k),
            ).fetchall()
        return [
            SearchResult(chunk=self._row_to_chunk(row), similarity=1.0)
            for row in rows
        ]

    def all_chunks(self, project_path: str | None = None) -> list[CodeChunk]:
        with self._connect() as connection:
            if project_path is None:
                rows = connection.execute(
                    """
                    SELECT file_path, chunk_type, name, content, start_line, end_line, embedding_json
                    FROM code_chunks
                    ORDER BY id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT file_path, chunk_type, name, content, start_line, end_line, embedding_json
                    FROM code_chunks
                    WHERE project_path = ?
                    ORDER BY id
                    """,
                    (project_path,),
                ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def save_relations(self, relations: list[CodeRelation], project_path: str = "") -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO code_relations
                    (project_path, file_path, from_name, to_name, relation_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        project_path,
                        relation.file_path,
                        relation.source,
                        relation.target,
                        relation.relation_type,
                    )
                    for relation in relations
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def find_relations(
        self,
        project_path: str = "",
        source: str | None = None,
        target: str | None = None,
        relation_type: str | None = None,
    ) -> list[CodeRelation]:
        clauses = ["project_path = ?"]
        params: list[str] = [project_path]
        if source is not None:
            clauses.append("from_name = ?")
            params.append(source)
        if target is not None:
            clauses.append("to_name = ?")
            params.append(target)
        if relation_type is not None:
            clauses.append("relation_type = ?")
            params.append(relation_type)
        sql = (
            "SELECT from_name, to_name, relation_type, file_path "
            "FROM code_relations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            CodeRelation(
                source=row[0],
                target=row[1],
                relation_type=row[2],
                file_path=row[3],
            )
            for row in rows
        ]

    def clear(self, project_path: str | None = None) -> None:
        with self._connect() as connection:
            if project_path is None:
                connection.execute("DELETE FROM code_chunks")
                connection.execute("DELETE FROM code_relations")
            else:
                connection.execute("DELETE FROM code_chunks WHERE project_path = ?", (project_path,))
                connection.execute("DELETE FROM code_relations WHERE project_path = ?", (project_path,))

    def _initialize(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    from_name TEXT NOT NULL,
                    to_name TEXT NOT NULL,
                    relation_type TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_chunks_project_path ON code_chunks(project_path)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_chunks_file_path ON code_chunks(file_path)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_chunks_chunk_type ON code_chunks(chunk_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_relations_from_name ON code_relations(from_name)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_relations_to_name ON code_relations(to_name)"
            )

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(self.db_path)
            return self._memory_connection
        return sqlite3.connect(self.db_path)

    def _row_to_chunk(self, row: tuple) -> CodeChunk:
        return CodeChunk(
            file_path=row[0],
            type=ChunkType(row[1]),
            name=row[2],
            content=row[3],
            start_line=row[4],
            end_line=row[5],
            embedding=json.loads(row[6] or "[]"),
        )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        dot = sum(left[index] * right[index] for index in range(length))
        left_norm = math.sqrt(sum(value * value for value in left[:length]))
        right_norm = math.sqrt(sum(value * value for value in right[:length]))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
