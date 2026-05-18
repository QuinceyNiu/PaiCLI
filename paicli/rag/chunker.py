"""Code chunking for repository RAG indexing."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from paicli.rag.models import CodeChunk


class CodeChunker:
    PYTHON_SUFFIXES = {".py"}
    JAVA_SUFFIXES = {".java"}
    JAVA_CLASS_PATTERN = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)")
    JAVA_METHOD_PATTERN = re.compile(r"([A-Za-z_]\w*)\s*\(([^)]*)\)")
    JAVA_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "synchronized"}

    def __init__(
        self,
        project_root: str | Path | None = None,
        max_text_chunk_chars: int = 2_000,
        class_header_lines: int = 5,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        self.max_text_chunk_chars = max_text_chunk_chars
        self.class_header_lines = class_header_lines

    def chunk_file(self, file_path: str | Path) -> list[CodeChunk]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        relative_path = self._relative_path(path)

        if path.suffix in self.JAVA_SUFFIXES:
            try:
                return self._chunk_java_file(relative_path, content)
            except ValueError:
                return self._chunk_large_text(relative_path, content)

        if path.suffix not in self.PYTHON_SUFFIXES:
            return self._chunk_large_text(relative_path, content)

        try:
            return self._chunk_python_file(relative_path, content)
        except SyntaxError:
            return self._chunk_large_text(relative_path, content)

    def _chunk_python_file(self, relative_path: str, content: str) -> list[CodeChunk]:
        tree = ast.parse(content)
        lines = content.splitlines()
        chunks: list[CodeChunk] = []

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            class_start = node.lineno
            class_end = min(
                getattr(node, "end_lineno", class_start),
                class_start + self.class_header_lines - 1,
            )
            chunks.append(
                CodeChunk.class_chunk(
                    file_path=relative_path,
                    name=node.name,
                    content=self._slice_lines(lines, class_start, class_end),
                    start_line=class_start,
                    end_line=class_end,
                )
            )

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_start = child.lineno
                    method_end = getattr(child, "end_lineno", child.lineno)
                    chunks.append(
                        CodeChunk.method_chunk(
                            file_path=relative_path,
                            name=f"{node.name}.{child.name}",
                            content=self._slice_lines(lines, method_start, method_end),
                            start_line=method_start,
                            end_line=method_end,
                        )
                    )

        if chunks:
            return chunks
        return self._chunk_large_text(relative_path, content)

    def _chunk_java_file(self, relative_path: str, content: str) -> list[CodeChunk]:
        lines = content.splitlines()
        chunks: list[CodeChunk] = []

        for index, line in enumerate(lines):
            match = self.JAVA_CLASS_PATTERN.search(line)
            if match is None:
                continue

            class_name = match.group(1)
            class_start = index + 1
            class_end = self._find_java_block_end(lines, index)
            class_header_end = min(class_end, class_start + self.class_header_lines - 1)
            chunks.append(
                CodeChunk.class_chunk(
                    file_path=relative_path,
                    name=class_name,
                    content=self._slice_lines(lines, class_start, class_header_end),
                    start_line=class_start,
                    end_line=class_header_end,
                )
            )

            chunks.extend(
                self._chunk_java_methods(
                    relative_path=relative_path,
                    class_name=class_name,
                    lines=lines,
                    body_start_index=index + 1,
                    body_end_index=class_end - 1,
                )
            )

        if not chunks:
            raise ValueError("No Java classes found")
        return chunks

    def _chunk_java_methods(
        self,
        relative_path: str,
        class_name: str,
        lines: list[str],
        body_start_index: int,
        body_end_index: int,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        index = body_start_index
        while index < body_end_index:
            line = lines[index]
            stripped = line.strip()
            if self._looks_like_java_method(stripped):
                method_end = self._find_java_block_end(lines, index)
                signature = stripped.split("{", 1)[0].strip()
                name_match = self.JAVA_METHOD_PATTERN.search(signature)
                if name_match is not None:
                    method_name = name_match.group(1)
                    parameters = name_match.group(2).strip()
                    chunks.append(
                        CodeChunk.method_chunk(
                            file_path=relative_path,
                            name=f"{class_name}.{method_name}({parameters})",
                            content=self._slice_lines(lines, index + 1, method_end),
                            start_line=index + 1,
                            end_line=method_end,
                        )
                    )
                    index = method_end
                    continue
            index += 1
        return chunks

    def _looks_like_java_method(self, stripped_line: str) -> bool:
        if "{" not in stripped_line or "(" not in stripped_line or ")" not in stripped_line:
            return False
        if stripped_line.startswith(("@", "//", "/*", "*")):
            return False
        name_match = self.JAVA_METHOD_PATTERN.search(stripped_line)
        if name_match is None:
            return False
        return name_match.group(1) not in self.JAVA_CONTROL_KEYWORDS

    def _find_java_block_end(self, lines: list[str], start_index: int) -> int:
        balance = 0
        seen_open = False
        for index in range(start_index, len(lines)):
            line = self._strip_java_line_comment(lines[index])
            balance += line.count("{")
            if "{" in line:
                seen_open = True
            balance -= line.count("}")
            if seen_open and balance == 0:
                return index + 1
        raise ValueError("Unbalanced Java block")

    def _strip_java_line_comment(self, line: str) -> str:
        return line.split("//", 1)[0]

    def _chunk_large_text(self, relative_path: str, content: str) -> list[CodeChunk]:
        if not content:
            return [
                CodeChunk.file_chunk(
                    file_path=relative_path,
                    name=f"{relative_path}#1",
                    content="",
                    start_line=1,
                    end_line=1,
                )
            ]

        chunks: list[CodeChunk] = []
        start = 0
        chunk_index = 1
        while start < len(content):
            end = min(len(content), start + self.max_text_chunk_chars)
            if end < len(content):
                newline = content.rfind("\n", start, end)
                if newline > start:
                    end = newline + 1
            chunk_content = content[start:end]
            start_line = content.count("\n", 0, start) + 1
            end_line = start_line + chunk_content.count("\n")
            if chunk_content.endswith("\n") and end_line > start_line:
                end_line -= 1
            chunks.append(
                CodeChunk.file_chunk(
                    file_path=relative_path,
                    name=f"{relative_path}#{chunk_index}",
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            start = end
            chunk_index += 1
        return chunks

    def _relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        if self.project_root is not None:
            try:
                return resolved.relative_to(self.project_root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def _slice_lines(self, lines: list[str], start_line: int, end_line: int) -> str:
        return "\n".join(lines[start_line - 1 : end_line])
