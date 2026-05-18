"""Hybrid retrieval over code chunks."""

from __future__ import annotations

import re
from collections import OrderedDict

from paicli.rag.embedding import EmbeddingClient
from paicli.rag.models import ChunkType, CodeChunk, SearchResult
from paicli.rag.store import VectorStore


class RagQueryTokenizer:
    TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\w\u4e00-\u9fff]+", re.UNICODE)
    SYNONYMS = {
        "压缩": ("compress", "compression"),
        "上下文": ("context",),
        "记忆": ("memory",),
        "检索": ("retrieve", "search"),
        "搜索": ("search",),
        "索引": ("index",),
        "工具": ("tool",),
        "调用": ("call",),
        "关系": ("relation",),
        "向量": ("embedding", "vector"),
    }
    STOP_WORDS = {
        "一下",
        "什么",
        "代码",
        "实现",
        "如何",
        "怎么",
        "方法",
        "用户",
        "这个",
        "那个",
        "里面",
        "the",
        "and",
        "for",
    }

    @classmethod
    def tokenize(cls, query: str) -> list[str]:
        raw_tokens = cls._segment(query)
        tokens: OrderedDict[str, None] = OrderedDict()
        for token in raw_tokens:
            for part in cls._split_identifier(token):
                cleaned = part.strip()
                if cls._is_valuable(cleaned):
                    tokens[cleaned] = None
                for source, expansions in cls.SYNONYMS.items():
                    if source in cleaned:
                        for expansion in expansions:
                            tokens[expansion] = None
        return list(tokens.keys())

    @classmethod
    def _segment(cls, query: str) -> list[str]:
        try:
            import jieba  # type: ignore[import-not-found]
        except Exception:
            tokens = cls.TOKEN_PATTERN.findall(query)
            expanded: list[str] = []
            for token in tokens:
                expanded.append(token)
                if cls._is_chinese(token) and len(token) > 2:
                    expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
            return expanded
        return [str(token) for token in jieba.cut(query)]

    @classmethod
    def _split_identifier(cls, token: str) -> list[str]:
        return [part for part in re.split(r"[.\s:/#()]+", token) if part]

    @classmethod
    def _is_valuable(cls, token: str) -> bool:
        if len(token) <= 1:
            return False
        return token not in cls.STOP_WORDS and token.lower() not in cls.STOP_WORDS

    @staticmethod
    def _is_chinese(token: str) -> bool:
        return all("\u4e00" <= char <= "\u9fff" for char in token)


class CodeRetriever:
    TYPE_BOOSTS = {
        ChunkType.METHOD: 0.15,
        ChunkType.CLASS: 0.10,
        ChunkType.FILE: 0.0,
    }
    DOUBLE_HIT_BONUS = 0.10
    TEST_PATH_PENALTY = 1.80
    TEST_QUERY_TERMS = {"test", "tests", "testing", "unittest", "pytest", "测试", "单测"}

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client or EmbeddingClient()

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        project_path: str = "",
        max_per_file: int = 2,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_client.embed(query)
        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=max(top_k * 4, top_k),
            project_path=project_path,
        )
        tokens = RagQueryTokenizer.tokenize(query)
        penalize_tests = not self._query_mentions_tests(tokens)
        explicit_identifiers = self._explicit_identifiers(tokens)

        merged: dict[str, SearchResult] = {}
        semantic_keys: set[str] = set()
        for result in semantic_results:
            key = self._chunk_key(result.chunk)
            semantic_keys.add(key)
            merged[key] = SearchResult(
                chunk=result.chunk,
                similarity=(
                    result.similarity
                    + self._type_boost(result.chunk)
                    + self._identifier_adjustment(result.chunk, explicit_identifiers)
                    - self._path_penalty(result.chunk, penalize_tests)
                ),
            )

        keyword_hits: set[str] = set()
        for chunk in self.vector_store.all_chunks(project_path=project_path):
            keyword_score = self._keyword_score(chunk, tokens)
            if keyword_score <= 0:
                continue
            key = self._chunk_key(chunk)
            keyword_hits.add(key)
            existing = merged.get(key)
            base_score = (
                existing.similarity
                if existing is not None
                else (
                    self._type_boost(chunk)
                    + self._identifier_adjustment(chunk, explicit_identifiers)
                    - self._path_penalty(chunk, penalize_tests)
                )
            )
            merged[key] = SearchResult(
                chunk=chunk,
                similarity=base_score + keyword_score,
            )

        for key in semantic_keys & keyword_hits:
            result = merged[key]
            merged[key] = SearchResult(
                chunk=result.chunk,
                similarity=result.similarity + self.DOUBLE_HIT_BONUS,
            )

        sorted_results = sorted(
            merged.values(),
            key=lambda result: result.similarity,
            reverse=True,
        )
        return self._limit_per_file(sorted_results, top_k=top_k, max_per_file=max_per_file)

    def _keyword_score(self, chunk: CodeChunk, tokens: list[str]) -> float:
        score = 0.0
        for token in tokens:
            lowered = token.lower()
            if lowered in chunk.name.lower():
                score += 0.30
            if lowered in chunk.file_path.lower():
                score += 0.10
            if lowered in chunk.content.lower():
                score += 0.10
        return score

    def _type_boost(self, chunk: CodeChunk) -> float:
        return self.TYPE_BOOSTS.get(chunk.type, 0.0)

    def _path_penalty(self, chunk: CodeChunk, penalize_tests: bool) -> float:
        if not penalize_tests:
            return 0.0
        normalized = chunk.file_path.replace("\\", "/").lower()
        if "/tests/" in f"/{normalized}" or normalized.startswith("tests/") or "/test_" in normalized:
            return self.TEST_PATH_PENALTY
        return 0.0

    def _query_mentions_tests(self, tokens: list[str]) -> bool:
        return any(token.lower() in self.TEST_QUERY_TERMS for token in tokens)

    def _explicit_identifiers(self, tokens: list[str]) -> list[str]:
        identifiers = []
        for token in tokens:
            if re.search(r"[A-Z][a-z0-9]+[A-Z_]?", token) or "_" in token or "." in token:
                identifiers.append(token.lower())
        return identifiers

    def _identifier_adjustment(self, chunk: CodeChunk, identifiers: list[str]) -> float:
        if not identifiers:
            return 0.0
        name_and_path = " ".join([chunk.name, chunk.file_path]).lower()
        content = chunk.content.lower()
        strong_matches = sum(1 for identifier in identifiers if identifier in name_and_path)
        weak_matches = sum(
            1
            for identifier in identifiers
            if identifier not in name_and_path and identifier in content
        )
        if strong_matches or weak_matches:
            return 0.80 * strong_matches + 0.15 * weak_matches
        return -0.25

    def _limit_per_file(
        self,
        sorted_results: list[SearchResult],
        top_k: int,
        max_per_file: int,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        file_counts: dict[str, int] = {}
        for result in sorted_results:
            count = file_counts.get(result.chunk.file_path, 0)
            if count >= max_per_file:
                continue
            results.append(result)
            file_counts[result.chunk.file_path] = count + 1
            if len(results) >= top_k:
                break
        return results

    def _chunk_key(self, chunk: CodeChunk) -> str:
        return f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.name}"
