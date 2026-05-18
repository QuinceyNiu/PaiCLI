"""Repository indexing pipeline for code RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paicli.rag.analyzer import CodeAnalyzer
from paicli.rag.chunker import CodeChunker
from paicli.rag.embedding import EmbeddingClient
from paicli.rag.models import CodeChunk
from paicli.rag.store import VectorStore


@dataclass(frozen=True)
class IndexProgress:
    processed_files: int
    total_files: int
    file_path: str


@dataclass(frozen=True)
class IndexStats:
    project_path: str
    file_count: int
    chunk_count: int
    relation_count: int
    failed_files: int = 0


class CodeIndex:
    SKIPPED_DIRS = {
        ".git",
        ".idea",
        ".mvn",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
    }
    SUPPORTED_SUFFIXES = {
        ".java",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".xml",
        ".txt",
    }

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: CodeChunker | None = None,
        analyzer: CodeAnalyzer | None = None,
        embedding_client: EmbeddingClient | None = None,
        on_progress: Callable[[IndexProgress], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chunker = chunker
        self.analyzer = analyzer
        self.embedding_client = embedding_client or EmbeddingClient()
        self.on_progress = on_progress
        self.on_warning = on_warning

    def index(self, project_root: str | Path) -> int:
        return self.index_with_stats(project_root).chunk_count

    def index_with_stats(self, project_root: str | Path) -> IndexStats:
        root = Path(project_root).resolve()
        project_path = str(root)
        chunker = self.chunker or CodeChunker(project_root=root)
        analyzer = self.analyzer or CodeAnalyzer(project_root=root)
        files = self._supported_files(root)
        self.vector_store.clear(project_path=project_path)
        indexed_count = 0
        relation_count = 0
        failed_files = 0

        for index, path in enumerate(files, start=1):
            try:
                chunks = [
                    self._embed_chunk(chunk)
                    for chunk in chunker.chunk_file(path)
                ]
                if chunks:
                    self.vector_store.save_chunks(chunks, project_path=project_path)
                    indexed_count += len(chunks)
                relations = analyzer.analyze_file(path)
                if relations:
                    self.vector_store.save_relations(relations, project_path=project_path)
                    relation_count += len(relations)
            except Exception as exc:
                failed_files += 1
                if self.on_warning is not None:
                    self.on_warning(f"索引文件失败: {path} ({exc})")
            if self.on_progress is not None and (index % 10 == 0 or index == len(files)):
                self.on_progress(
                    IndexProgress(
                        processed_files=index,
                        total_files=len(files),
                        file_path=path.name,
                    )
                )

        return IndexStats(
            project_path=project_path,
            file_count=len(files),
            chunk_count=indexed_count,
            relation_count=relation_count,
            failed_files=failed_files,
        )

    def _embed_chunk(self, chunk: CodeChunk) -> CodeChunk:
        return chunk.with_embedding(self.embedding_client.embed(chunk.to_embedding_text()))

    def _supported_files(self, root: Path) -> list[Path]:
        paths: list[Path] = []
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in self.SKIPPED_DIRS
            ]
            for filename in files:
                path = Path(current_root) / filename
                if path.suffix in self.SUPPORTED_SUFFIXES:
                    paths.append(path)
        return paths
