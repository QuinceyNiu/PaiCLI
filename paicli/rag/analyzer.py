"""Code relationship analysis for RAG graph queries."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from paicli.rag.models import CodeRelation


class CodeAnalyzer:
    JAVA_IMPORT_PATTERN = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)(?:\.\*)?\s*;")
    JAVA_CLASS_PATTERN = re.compile(
        r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)"
        r"(?:\s+extends\s+([A-Za-z_]\w*))?"
        r"(?:\s+implements\s+([A-Za-z_][\w\s,<>.]*))?"
    )
    JAVA_METHOD_PATTERN = re.compile(r"([A-Za-z_]\w*)\s*\(([^)]*)\)")
    JAVA_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    JAVA_CALL_EXCLUDES = {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "new",
        "super",
        "this",
    }
    STDLIB_IMPORT_ROOTS = {
        "abc",
        "ast",
        "collections",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "re",
        "sqlite3",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
        "unittest",
        "uuid",
    }

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None

    def analyze_file(self, file_path: str | Path) -> list[CodeRelation]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        relative_path = self._relative_path(path)
        if path.suffix == ".java":
            return self._analyze_java(relative_path, content)
        if path.suffix == ".py":
            return self._analyze_python(relative_path, content)
        return []

    def _analyze_java(self, file_path: str, content: str) -> list[CodeRelation]:
        lines = content.splitlines()
        relations: list[CodeRelation] = []

        for line in lines:
            match = self.JAVA_IMPORT_PATTERN.match(line)
            if match is None:
                continue
            imported = match.group(1)
            if not imported.startswith(("java.", "javax.")):
                relations.append(CodeRelation(file_path, imported, "imports", file_path))

        for index, line in enumerate(lines):
            class_match = self.JAVA_CLASS_PATTERN.search(line)
            if class_match is None:
                continue
            class_name = class_match.group(1)
            extended = class_match.group(2)
            implemented = class_match.group(3)
            if extended:
                relations.append(CodeRelation(class_name, extended, "extends", file_path))
            if implemented:
                for interface in implemented.split(","):
                    cleaned = interface.strip().split("<", 1)[0].strip()
                    if cleaned:
                        relations.append(CodeRelation(class_name, cleaned, "implements", file_path))

            class_end = self._find_java_block_end(lines, index)
            relations.extend(
                self._java_method_relations(file_path, class_name, lines, index + 1, class_end - 1)
            )

        return relations

    def _java_method_relations(
        self,
        file_path: str,
        class_name: str,
        lines: list[str],
        start_index: int,
        end_index: int,
    ) -> list[CodeRelation]:
        relations: list[CodeRelation] = []
        index = start_index
        while index < end_index:
            stripped = lines[index].strip()
            if self._looks_like_java_method(stripped):
                method_match = self.JAVA_METHOD_PATTERN.search(stripped)
                if method_match is None:
                    index += 1
                    continue
                method_name = method_match.group(1)
                full_method = f"{class_name}.{method_name}"
                relations.append(CodeRelation(class_name, full_method, "contains", file_path))
                method_end = self._find_java_block_end(lines, index)
                body = "\n".join(lines[index + 1 : method_end])
                for call in self.JAVA_CALL_PATTERN.findall(body):
                    if call not in self.JAVA_CALL_EXCLUDES:
                        relations.append(CodeRelation(full_method, call, "calls", file_path))
                index = method_end
                continue
            index += 1
        return relations

    def _analyze_python(self, file_path: str, content: str) -> list[CodeRelation]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        relations: list[CodeRelation] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_stdlib_import(alias.name):
                        relations.append(CodeRelation(file_path, alias.name, "imports", file_path))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not self._is_stdlib_import(module):
                    for alias in node.names:
                        relations.append(CodeRelation(file_path, f"{module}.{alias.name}", "imports", file_path))
            elif isinstance(node, ast.ClassDef):
                relations.extend(self._python_class_relations(file_path, node))
        return relations

    def _python_class_relations(self, file_path: str, node: ast.ClassDef) -> list[CodeRelation]:
        relations: list[CodeRelation] = []
        class_name = node.name
        for base in node.bases:
            base_name = self._python_name(base)
            if base_name:
                relations.append(CodeRelation(class_name, base_name, "extends", file_path))
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = f"{class_name}.{child.name}"
                relations.append(CodeRelation(class_name, method_name, "contains", file_path))
                for descendant in ast.walk(child):
                    if isinstance(descendant, ast.Call):
                        call_name = self._python_call_name(descendant.func)
                        if call_name:
                            relations.append(CodeRelation(method_name, call_name, "calls", file_path))
        return relations

    def _python_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _python_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._python_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def _looks_like_java_method(self, stripped_line: str) -> bool:
        if "{" not in stripped_line or "(" not in stripped_line or ")" not in stripped_line:
            return False
        if stripped_line.startswith(("@", "//", "/*", "*")):
            return False
        match = self.JAVA_METHOD_PATTERN.search(stripped_line)
        return match is not None and match.group(1) not in self.JAVA_CALL_EXCLUDES

    def _find_java_block_end(self, lines: list[str], start_index: int) -> int:
        balance = 0
        seen_open = False
        for index in range(start_index, len(lines)):
            line = lines[index].split("//", 1)[0]
            balance += line.count("{")
            if "{" in line:
                seen_open = True
            balance -= line.count("}")
            if seen_open and balance == 0:
                return index + 1
        return len(lines)

    def _is_stdlib_import(self, module: str) -> bool:
        return module.split(".", 1)[0] in self.STDLIB_IMPORT_ROOTS

    def _relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        if self.project_root is not None:
            try:
                return resolved.relative_to(self.project_root).as_posix()
            except ValueError:
                pass
        return path.as_posix()
