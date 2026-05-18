"""Memory manager facade and its first-pass component boundaries."""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from paicli.llm.glm_client import Message
from paicli.memory.entry import MemoryEntry, MemoryType, estimate_tokens


class ConversationMemory:
    def __init__(self, max_tokens: int = 16_000) -> None:
        self.max_tokens = max_tokens
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.current_tokens = 0
        self.compressed_summaries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.store(entry)

    def store(self, entry: MemoryEntry) -> None:
        existing = self._entries.pop(entry.id, None)
        if existing is not None:
            self.current_tokens -= existing.token_count
        self._entries[entry.id] = entry
        self.current_tokens += entry.token_count

        while self.current_tokens > self.max_tokens and len(self._entries) > 1:
            self._evict_oldest()

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def get_all(self) -> list[MemoryEntry]:
        return list(self.compressed_summaries) + self.entries()

    def clear(self) -> None:
        self._entries.clear()
        self.current_tokens = 0
        self.compressed_summaries.clear()

    def get_usage_ratio(self) -> float:
        if self.max_tokens <= 0:
            return 1.0
        return self.current_tokens / self.max_tokens

    def get_status_summary(self) -> str:
        return (
            f"短期记忆: {len(self.entries())}条 / "
            f"{self.current_tokens} tokens "
            f"(预算: {self.max_tokens}, "
            f"使用率: {self.get_usage_ratio():.0%}, "
            f"已压缩: {len(self.compressed_summaries)}条)"
        )

    def _evict_oldest(self) -> None:
        _, oldest = self._entries.popitem(last=False)
        self.current_tokens -= oldest.token_count
        self.compressed_summaries.append(oldest)


class LongTermMemory:
    STORAGE_FILE = "long_term_memory.json"

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else self._default_storage_dir()
        self.storage_file = self.storage_dir / self.STORAGE_FILE
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.current_tokens = 0
        self.load_from_disk()

    def add(self, entry: MemoryEntry) -> None:
        self.store(entry)

    def store(self, entry: MemoryEntry) -> MemoryEntry | None:
        if any(existing.content == entry.content for existing in self._entries.values()):
            return None
        self._entries[entry.id] = entry
        self.current_tokens += entry.token_count
        self.save_to_disk()
        return entry

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def get_status_summary(self) -> str:
        facts = sum(1 for entry in self.entries() if entry.type == MemoryType.FACT)
        summaries = sum(1 for entry in self.entries() if entry.type == MemoryType.SUMMARY)
        tool_results = sum(1 for entry in self.entries() if entry.type == MemoryType.TOOL_RESULT)
        return (
            f"长期记忆: {len(self.entries())}条 / {self.current_tokens} tokens "
            f"(事实: {facts}, 摘要: {summaries}, 工具结果: {tool_results})"
        )

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        query_tokens = MemoryQueryTokenizer.tokenize(query)
        if not query_tokens:
            return self.entries()[:limit]

        matches = []
        for entry in self._entries.values():
            if MemoryQueryTokenizer.matches(entry.content, query_tokens):
                matches.append(entry)
            elif any(
                MemoryQueryTokenizer.matches(value, query_tokens)
                for value in entry.metadata.values()
            ):
                matches.append(entry)
            if len(matches) >= limit:
                break
        return matches

    def save_to_disk(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = [entry.to_dict() for entry in self._entries.values()]
        self.storage_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_from_disk(self) -> None:
        if not self.storage_file.exists():
            return
        data = json.loads(self.storage_file.read_text(encoding="utf-8"))
        self._entries.clear()
        self.current_tokens = 0
        for raw_entry in data:
            entry = MemoryEntry.from_dict(raw_entry)
            self._entries[entry.id] = entry
            self.current_tokens += entry.token_count

    def _default_storage_dir(self) -> Path:
        configured = os.getenv("PAICLI_MEMORY_DIR")
        if configured:
            return Path(configured)
        return Path.home() / ".paicli" / "memory"


class ContextCompressor:
    MAP_PROMPT = """
请把下面这组历史对话压缩成简洁摘要，保留用户偏好、项目约定、工具结果和关键决策：

{conversation}
""".strip()
    REDUCE_PROMPT = """
请合并下面多个历史对话分片摘要，去重后生成一段最终摘要：

{summaries}
""".strip()
    EXTRACT_FACTS_PROMPT = """
请从下面对话中提取可跨会话复用的关键事实。
只输出事实列表，每行一条，不要解释。
关注用户偏好、项目配置、技术栈和重要决策。

{conversation}
""".strip()

    def __init__(
        self,
        llm_client: Any | None = None,
        retain_recent_rounds: int = 3,
        chunk_size: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.retain_recent_rounds = retain_recent_rounds
        self.chunk_size = chunk_size

    def compress(self, memory: ConversationMemory | list[MemoryEntry]) -> str | list[MemoryEntry]:
        if isinstance(memory, list):
            return memory

        all_entries = memory.get_all()
        if len(all_entries) <= self.retain_recent_rounds:
            return ""

        split_point = len(all_entries) - self.retain_recent_rounds
        old_entries = list(all_entries[:split_point])
        recent_entries = list(all_entries[split_point:])

        chunk_summaries = self._map_phase(old_entries)
        if not chunk_summaries:
            return ""
        final_summary = (
            chunk_summaries[0]
            if len(chunk_summaries) == 1
            else self._reduce_phase(chunk_summaries)
        )

        memory.clear()
        memory.store(
            MemoryEntry(
                id=f"summary-{uuid4().hex[:8]}",
                content=f"[历史对话摘要] {final_summary}",
                type=MemoryType.SUMMARY,
            )
        )
        for entry in recent_entries:
            memory.store(entry)
        return final_summary

    def extract_facts(
        self,
        entries: list[MemoryEntry],
        long_term_memory: LongTermMemory,
    ) -> list[str]:
        if not entries:
            return []

        conversation = self._format_entries(entries)
        prompt = self.EXTRACT_FACTS_PROMPT.format(conversation=conversation)
        response_text = self._chat(prompt)
        facts = self._parse_facts(response_text)
        for fact in facts:
            long_term_memory.store(
                MemoryEntry(
                    id=f"fact-{uuid4().hex[:8]}",
                    content=fact,
                    type=MemoryType.FACT,
                )
            )
        return facts

    def _map_phase(self, entries: list[MemoryEntry]) -> list[str]:
        summaries = []
        for index in range(0, len(entries), self.chunk_size):
            chunk = entries[index : index + self.chunk_size]
            prompt = self.MAP_PROMPT.format(conversation=self._format_entries(chunk))
            summaries.append(self._chat(prompt))
        return summaries

    def _reduce_phase(self, summaries: list[str]) -> str:
        prompt = self.REDUCE_PROMPT.format(summaries="\n".join(summaries))
        return self._chat(prompt)

    def _chat(self, prompt: str) -> str:
        if self.llm_client is None:
            return self._fallback_summary(prompt)
        response = self.llm_client.chat([Message.user(prompt)], [])
        return response.message.content or ""

    def _fallback_summary(self, prompt: str) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        return "；".join(lines[-self.chunk_size :])

    def _format_entries(self, entries: list[MemoryEntry]) -> str:
        return "\n".join(
            f"[{entry.type.value}] {entry.timestamp.isoformat()} {entry.content}"
            for entry in entries
        )

    def _parse_facts(self, text: str) -> list[str]:
        facts = []
        for line in text.splitlines():
            fact = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", line).strip()
            if fact and fact not in {"无", "没有", "无事实"}:
                facts.append(fact)
        return facts


class TokenBudget:
    def __init__(
        self,
        context_window: int = 200_000,
        reserved_for_system: int = 500,
        reserved_for_tools: int = 800,
        reserved_for_response: int = 2_000,
    ) -> None:
        self.context_window = context_window
        self.reserved_for_system = reserved_for_system
        self.reserved_for_tools = reserved_for_tools
        self.reserved_for_response = reserved_for_response
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.llm_call_count = 0

    def get_available_for_conversation(self) -> int:
        return max(
            0,
            self.context_window
            - self.reserved_for_system
            - self.reserved_for_tools
            - self.reserved_for_response,
        )

    def needs_compression(self, memory: ConversationMemory) -> bool:
        return memory.current_tokens > self.get_available_for_conversation() * 0.8

    def enforce(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        return entries

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_call_count += 1

    def get_usage_report(self) -> str:
        average_input = (
            self.total_input_tokens // self.llm_call_count
            if self.llm_call_count
            else 0
        )
        return (
            f"Token 统计: 调用 {self.llm_call_count} 次 | "
            f"总输入: {self.total_input_tokens} | "
            f"总输出: {self.total_output_tokens} | "
            f"平均输入: {average_input} | "
            f"预算: {self.context_window} "
            f"(可用: {self.get_available_for_conversation()})"
        )


@dataclass(frozen=True)
class ScoredEntry:
    entry: MemoryEntry
    score: float


class MemoryRetriever:
    def __init__(self, now_func: Callable[[], datetime] | None = None) -> None:
        self.now_func = now_func or (lambda: datetime.now(timezone.utc))

    def retrieve(
        self,
        query: str,
        short_term: ConversationMemory | list[MemoryEntry],
        long_term: LongTermMemory | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        if not query:
            if isinstance(short_term, list):
                return short_term[:limit]
            entries = short_term.get_all()
            if long_term is not None:
                entries += long_term.entries()
            return entries[:limit]

        query_tokens = MemoryQueryTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scored: list[ScoredEntry] = []
        if isinstance(short_term, list):
            for entry in short_term:
                score = self.compute_relevance_score(entry, query_tokens)
                if score > 0:
                    scored.append(ScoredEntry(entry, score))
        else:
            for entry in short_term.get_all():
                score = self.compute_relevance_score(entry, query_tokens)
                if score > 0:
                    scored.append(ScoredEntry(entry, score))
            if long_term is not None:
                for entry in long_term.entries():
                    score = self.compute_relevance_score(entry, query_tokens) * 1.2
                    if score > 0:
                        scored.append(ScoredEntry(entry, score))

        return [
            scored_entry.entry
            for scored_entry in sorted(scored, key=lambda item: item.score, reverse=True)[:limit]
        ]

    def compute_relevance_score(
        self,
        entry: MemoryEntry,
        query_tokens: set[str],
    ) -> float:
        searchable = " ".join([entry.content, *entry.metadata.values()])
        matched = sum(
            1 for token in query_tokens if MemoryQueryTokenizer.matches(searchable, {token})
        )
        keyword_score = matched / len(query_tokens) if query_tokens else 0.0
        if keyword_score <= 0:
            return 0.0
        return keyword_score * self._time_decay(entry)

    def _time_decay(self, entry: MemoryEntry) -> float:
        age_seconds = max(0.0, (self.now_func() - entry.timestamp).total_seconds())
        if age_seconds <= 24 * 60 * 60:
            return 1.0 - 0.5 * (age_seconds / (24 * 60 * 60))
        return 0.5


class MemoryManager:
    def __init__(
        self,
        conversation_memory: ConversationMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        context_compressor: ContextCompressor | None = None,
        token_budget: TokenBudget | None = None,
        memory_retriever: MemoryRetriever | None = None,
    ) -> None:
        self.token_budget = token_budget or TokenBudget()
        self.conversation_memory = conversation_memory or ConversationMemory(
            max_tokens=self.token_budget.get_available_for_conversation()
        )
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.context_compressor = context_compressor or ContextCompressor()
        self.memory_retriever = memory_retriever or MemoryRetriever()

    def store_message(
        self,
        content: str,
        type: MemoryType = MemoryType.CONVERSATION,
        metadata: dict[str, str] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=f"mem_{uuid4().hex}",
            content=content,
            type=type,
            metadata=metadata or {},
        )
        if type == MemoryType.FACT:
            self.long_term_memory.store(entry)
        else:
            self.conversation_memory.store(entry)
        self._enforce_budget()
        return entry

    def add_user_message(self, content: str) -> MemoryEntry:
        return self.store_message(content, MemoryType.CONVERSATION, {"role": "user"})

    def add_assistant_message(self, content: str) -> MemoryEntry:
        return self.store_message(content, MemoryType.CONVERSATION, {"role": "assistant"})

    def add_tool_result(self, tool_name: str, result: str) -> MemoryEntry:
        return self.store_message(result, MemoryType.TOOL_RESULT, {"tool": tool_name})

    def save_fact(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=f"fact-{uuid4().hex[:8]}",
            content=content,
            type=MemoryType.FACT,
            metadata=metadata or {},
        )
        self.long_term_memory.store(entry)
        return entry

    def extract_and_save_facts(self) -> list[str]:
        return self.context_compressor.extract_facts(
            self.conversation_memory.get_all(),
            self.long_term_memory,
        )

    def clear_short_term(self) -> None:
        self.conversation_memory.clear()

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.token_budget.record_usage(input_tokens, output_tokens)

    def get_memory(self, query: str = "", limit: int = 5) -> list[MemoryEntry]:
        return self.memory_retriever.retrieve(
            query,
            self.conversation_memory,
            self.long_term_memory,
            limit=limit,
        )

    def build_context_for_query(
        self,
        query: str,
        max_tokens: int = 500,
        limit: int | None = None,
    ) -> str:
        memories = self.get_memory(query, limit=limit or 50)
        if not memories:
            return ""
        lines = ["[相关记忆]"]
        used_tokens = 0
        for entry in memories:
            line = f"- {entry.type.value}: {entry.content}"
            line_tokens = estimate_tokens(line)
            if used_tokens + line_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(line)
            used_tokens += line_tokens
        return "\n".join(lines)

    def get_system_status(self) -> str:
        return "\n".join(
            [
                self.conversation_memory.get_status_summary(),
                self.long_term_memory.get_status_summary(),
                self.token_budget.get_usage_report(),
            ]
        )

    def compress_if_needed(self) -> None:
        if self.token_budget.needs_compression(self.conversation_memory):
            self.context_compressor.compress(self.conversation_memory)
            self.token_budget.enforce(self.conversation_memory.entries())

    def _enforce_budget(self) -> None:
        self.compress_if_needed()


class MemoryQueryTokenizer:
    TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

    @classmethod
    def tokenize(cls, query: str) -> set[str]:
        normalized = query.lower().strip()
        if not normalized:
            return set()

        words = cls._segment(normalized)
        tokens: OrderedDict[str, None] = OrderedDict()
        for word in words:
            trimmed = word.strip()
            if len(trimmed) >= 2 and not cls._is_punctuation(trimmed):
                tokens[trimmed] = None
        return set(tokens.keys())

    @classmethod
    def matches(cls, text: str, query_tokens: set[str]) -> bool:
        text_tokens = cls.tokenize(text)
        normalized = text.lower()
        return any(token in text_tokens or token in normalized for token in query_tokens)

    @classmethod
    def _segment(cls, text: str) -> list[str]:
        try:
            import jieba  # type: ignore[import-not-found]
        except Exception:
            words = []
            for token in cls.TOKEN_PATTERN.findall(text):
                words.append(token)
                if cls._is_chinese_text(token) and len(token) > 2:
                    words.extend(token[index : index + 2] for index in range(len(token) - 1))
            return words
        return [str(word) for word in jieba.cut(text)]

    @staticmethod
    def _is_punctuation(text: str) -> bool:
        return all(not char.isalnum() and not ("\u4e00" <= char <= "\u9fff") for char in text)

    @staticmethod
    def _is_chinese_text(text: str) -> bool:
        return all("\u4e00" <= char <= "\u9fff" for char in text)
