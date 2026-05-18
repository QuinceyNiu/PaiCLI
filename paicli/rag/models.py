"""Data models for code RAG chunks and relationships."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChunkType(str, Enum):
    FILE = "file"
    CLASS = "class"
    METHOD = "method"


@dataclass(frozen=True)
class CodeChunk:
    file_path: str
    type: ChunkType
    name: str
    content: str
    start_line: int
    end_line: int
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def file_chunk(
        cls,
        file_path: str,
        name: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> "CodeChunk":
        return cls(
            file_path=file_path,
            type=ChunkType.FILE,
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )

    @classmethod
    def class_chunk(
        cls,
        file_path: str,
        name: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> "CodeChunk":
        return cls(
            file_path=file_path,
            type=ChunkType.CLASS,
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )

    @classmethod
    def method_chunk(
        cls,
        file_path: str,
        name: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> "CodeChunk":
        return cls(
            file_path=file_path,
            type=ChunkType.METHOD,
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )

    def with_embedding(self, embedding: list[float]) -> "CodeChunk":
        return CodeChunk(
            file_path=self.file_path,
            type=self.type,
            name=self.name,
            content=self.content,
            start_line=self.start_line,
            end_line=self.end_line,
            embedding=list(embedding),
        )

    def to_embedding_text(self) -> str:
        return f"[{self.type.value}:{self.name}] {self.content}"


@dataclass(frozen=True)
class CodeRelation:
    source: str
    target: str
    relation_type: str
    file_path: str


@dataclass(frozen=True)
class SearchResult:
    chunk: CodeChunk
    similarity: float
