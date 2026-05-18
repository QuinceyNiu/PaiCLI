"""Built-in tool registry for the PaiCli agent."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from paicli.llm.glm_client import Tool as LLMTool
from paicli.rag import CodeRetriever, SearchResultFormatter, VectorStore


ToolExecutor = Callable[[Mapping[str, str]], str]
CodeRetrieverFactory = Callable[[], CodeRetriever]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: ToolExecutor

    def to_llm_tool(self) -> LLMTool:
        return LLMTool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolProvider(Protocol):
    def get_tools(self) -> list[RegisteredTool]:
        """Return tools that should be registered."""


def create_parameters(*params: tuple[str, str, str, bool]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param_type, description, is_required in params:
        properties[name] = {
            "type": param_type,
            "description": description,
        }
        if is_required:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class ToolRegistry:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        providers: Iterable[ToolProvider] | None = None,
        code_retriever_factory: CodeRetrieverFactory | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or ".").resolve()
        self.project_path = str(self.base_dir)
        self.code_retriever_factory = code_retriever_factory
        self.tools: dict[str, RegisteredTool] = {}
        self._register_file_tools()
        self._register_shell_tools()
        self._register_code_tools()

        for provider in providers or []:
            for tool in provider.get_tools():
                self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        self.tools[tool.name] = tool

    def set_project_path(self, project_path: str | Path) -> None:
        self.project_path = str(Path(project_path).expanduser().resolve())

    def get(self, name: str) -> RegisteredTool | None:
        return self.tools.get(name)

    def tools_for_llm(self) -> list[LLMTool]:
        return [tool.to_llm_tool() for tool in self.tools.values()]

    def execute(self, name: str, args: Mapping[str, str]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"未知工具: {name}"

        try:
            return tool.executor(args)
        except Exception as exc:  # Defensive boundary for LLM-facing tool output.
            return f"工具执行失败: {exc}"

    def _register_file_tools(self) -> None:
        self.register(
            RegisteredTool(
                name="read_file",
                description="读取文件内容，用于查看代码、配置文件等",
                parameters=create_parameters(("path", "string", "文件路径", True)),
                executor=self._read_file,
            )
        )
        self.register(
            RegisteredTool(
                name="write_file",
                description="写入文件内容，用于创建或修改代码、配置和文档",
                parameters=create_parameters(
                    ("path", "string", "文件路径", True),
                    ("content", "string", "文件内容", True),
                ),
                executor=self._write_file,
            )
        )
        self.register(
            RegisteredTool(
                name="list_dir",
                description="列出目录内容，用于浏览项目结构",
                parameters=create_parameters(("path", "string", "目录路径", True)),
                executor=self._list_dir,
            )
        )

    def _register_shell_tools(self) -> None:
        self.register(
            RegisteredTool(
                name="execute_command",
                description="执行Shell命令，用于编译、运行、测试、Git操作等",
                parameters=create_parameters(("command", "string", "要执行的命令", True)),
                executor=self._execute_command,
            )
        )

    def _register_code_tools(self) -> None:
        self.register(
            RegisteredTool(
                name="create_project",
                description="创建项目结构。files 参数是 JSON 对象，key 是相对文件路径，value 是文件内容",
                parameters=create_parameters(
                    ("path", "string", "项目根目录路径", True),
                    ("files", "string", "JSON对象，描述要创建的文件和内容", True),
                ),
                executor=self._create_project,
            )
        )
        self.register(
            RegisteredTool(
                name="search_code",
                description="语义检索代码库，根据自然语言描述查找相关代码块",
                parameters=create_parameters(
                    ("query", "string", "自然语言查询描述，例如'用户登录的实现'", True),
                    ("top_k", "integer", "返回结果数量（默认5）", False),
                ),
                executor=self._search_code,
            )
        )

    def _read_file(self, args: Mapping[str, str]) -> str:
        path = self._resolve(args.get("path", ""))
        try:
            return "文件内容:\n" + path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"读取文件失败: {exc}"

    def _write_file(self, args: Mapping[str, str]) -> str:
        path = self._resolve(args.get("path", ""))
        content = args.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"文件已写入: {path}"
        except Exception as exc:
            return f"写入文件失败: {exc}"

    def _list_dir(self, args: Mapping[str, str]) -> str:
        path = self._resolve(args.get("path", "."))
        try:
            entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            lines = [f"[DIR] {entry.name}" if entry.is_dir() else f"[FILE] {entry.name}" for entry in entries]
            return "目录内容:\n" + "\n".join(lines)
        except Exception as exc:
            return f"列出目录失败: {exc}"

    def _execute_command(self, args: Mapping[str, str]) -> str:
        command = args.get("command", "")
        if not command.strip():
            return "执行命令失败: command 不能为空"

        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=self.base_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
            return f"命令执行完成 (exit code: {completed.returncode})\n{completed.stdout}"
        except Exception as exc:
            return f"执行命令失败: {exc}"

    def _create_project(self, args: Mapping[str, str]) -> str:
        root = self._resolve(args.get("path", ""))
        files_text = args.get("files", "{}")
        try:
            files = json.loads(files_text)
            if not isinstance(files, dict):
                return "创建项目失败: files 必须是 JSON 对象"

            root.mkdir(parents=True, exist_ok=True)
            created: list[str] = []
            for relative_path, content in files.items():
                file_path = root / str(relative_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(str(content), encoding="utf-8")
                created.append(str(relative_path))

            return "项目已创建: {}\n文件:\n{}".format(root, "\n".join(created))
        except Exception as exc:
            return f"创建项目失败: {exc}"

    def _search_code(self, args: Mapping[str, str]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "检索代码失败: query 不能为空"
        try:
            top_k = int(args.get("top_k") or 5)
        except ValueError:
            top_k = 5
        top_k = max(1, min(top_k, 20))

        retriever = self._create_code_retriever()
        results = retriever.hybrid_search(
            query,
            top_k=top_k,
            project_path=self.project_path,
        )
        return SearchResultFormatter.format_for_tool(query, results)

    def _create_code_retriever(self) -> CodeRetriever:
        if self.code_retriever_factory is not None:
            return self.code_retriever_factory()
        return CodeRetriever(VectorStore.default())

    def _resolve(self, path: str) -> Path:
        target = Path(path).expanduser()
        if target.is_absolute():
            return target
        return (self.base_dir / target).resolve()
