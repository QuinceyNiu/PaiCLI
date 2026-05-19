"""Built-in tool registry for the PaiCli agent."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from paicli.llm.glm_client import Tool as LLMTool
from paicli.rag import CodeRetriever, SearchResultFormatter, VectorStore


ToolExecutor = Callable[[Mapping[str, str]], str]
CodeRetrieverFactory = Callable[[], CodeRetriever]
MAX_PARALLEL_TOOLS = 4
DEFAULT_TOOL_BATCH_TIMEOUT_SECONDS = 90.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
MAX_COMMAND_OUTPUT_CHARS = 8_000


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


@dataclass(frozen=True)
class ToolInvocation:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolExecutionResult:
    id: str
    name: str
    arguments_json: str
    result: str
    elapsed_millis: int
    timed_out: bool = False

    @classmethod
    def completed(
        cls,
        invocation: ToolInvocation,
        result: str,
        elapsed_millis: int,
    ) -> "ToolExecutionResult":
        return cls(
            invocation.id,
            invocation.name,
            invocation.arguments_json,
            result,
            elapsed_millis,
            False,
        )

    @classmethod
    def timed_out_result(
        cls,
        invocation: ToolInvocation,
        timeout_seconds: float,
    ) -> "ToolExecutionResult":
        return cls(
            invocation.id,
            invocation.name,
            invocation.arguments_json,
            f"工具执行超时（{timeout_seconds:g}秒），已取消",
            int(timeout_seconds * 1000),
            True,
        )

    @classmethod
    def failed(cls, invocation: ToolInvocation, message: str) -> "ToolExecutionResult":
        return cls.completed(invocation, f"工具执行失败: {message}", 0)


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
        tool_batch_timeout_seconds: float = DEFAULT_TOOL_BATCH_TIMEOUT_SECONDS,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self.base_dir = Path(base_dir or ".").resolve()
        self.project_path = str(self.base_dir)
        self.code_retriever_factory = code_retriever_factory
        self.tool_batch_timeout_seconds = tool_batch_timeout_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.tools: dict[str, RegisteredTool] = {}
        self._register_file_tools()
        self._register_shell_tools()
        self._register_code_tools()
        self._register_web_tools()

        for provider in providers or []:
            for tool in provider.get_tools():
                self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        self.tools[tool.name] = tool

    def unregister_by_prefix(self, prefix: str) -> None:
        for name in [name for name in self.tools if name.startswith(prefix)]:
            del self.tools[name]

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

    def execute_tool(self, name: str, arguments_json: str) -> str:
        try:
            parsed_args = json.loads(arguments_json or "{}")
            if not isinstance(parsed_args, dict):
                return "工具参数解析失败: arguments 必须是 JSON 对象"
            args = {
                str(key): value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in parsed_args.items()
            }
        except Exception as exc:
            return f"工具参数解析失败: {exc}"

        return self.execute(name, args)

    def execute_tools(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        if not invocations:
            return []

        if len(invocations) == 1:
            invocation = invocations[0]
            started_at = time.perf_counter()
            result = self.execute_tool(invocation.name, invocation.arguments_json)
            return [
                ToolExecutionResult.completed(
                    invocation,
                    result,
                    self._elapsed_millis(started_at),
                )
            ]

        results: list[ToolExecutionResult | None] = [None] * len(invocations)
        work_queue: queue.Queue[tuple[int, ToolInvocation]] = queue.Queue()
        for index, invocation in enumerate(invocations):
            work_queue.put((index, invocation))

        def worker() -> None:
            while True:
                try:
                    index, invocation = work_queue.get_nowait()
                except queue.Empty:
                    return
                started_at = time.perf_counter()
                try:
                    result = self.execute_tool(invocation.name, invocation.arguments_json)
                    results[index] = ToolExecutionResult.completed(
                        invocation,
                        result,
                        self._elapsed_millis(started_at),
                    )
                except Exception as exc:
                    results[index] = ToolExecutionResult.failed(invocation, str(exc))
                finally:
                    work_queue.task_done()

        threads = [
            threading.Thread(target=worker, name="paicli-tool-executor", daemon=True)
            for _ in range(min(len(invocations), MAX_PARALLEL_TOOLS))
        ]
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + self.tool_batch_timeout_seconds
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

        return [
            result
            if result is not None
            else ToolExecutionResult.timed_out_result(invocation, self.tool_batch_timeout_seconds)
            for result, invocation in zip(results, invocations)
        ]

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

    def _register_web_tools(self) -> None:
        from paicli.web import WebToolProvider

        for tool in WebToolProvider().get_tools():
            self.register(tool)

    def _read_file(self, args: Mapping[str, str]) -> str:
        path = self._resolve(args.get("path", ""))
        try:
            return "文件内容:\n" + path.read_text(encoding="utf-8")
        except Exception as exc:
            return self._format_path_error("读取文件失败", path, exc)

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
            return self._format_path_error("列出目录失败", path, exc)

    def _execute_command(self, args: Mapping[str, str]) -> str:
        command = args.get("command", "")
        if not command.strip():
            return "执行命令失败: command 不能为空"
        if self._is_disallowed_broad_scan(command):
            return (
                "禁止执行全盘扫描命令。请改用 read_file、list_dir、search_code，"
                "或将 find 限制在当前项目的具体子目录内。"
            )

        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=self.base_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.command_timeout_seconds,
                check=False,
            )
            output = self._truncate_command_output(completed.stdout or "")
            return f"命令执行完成 (exit code: {completed.returncode})\n{output}"
        except subprocess.TimeoutExpired:
            return f"命令执行超时（{self.command_timeout_seconds:g}秒），已强制终止"
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

    def _truncate_command_output(self, output: str) -> str:
        if len(output) <= MAX_COMMAND_OUTPUT_CHARS:
            return output
        return output[:MAX_COMMAND_OUTPUT_CHARS] + "\n...(输出已截断)"

    def _is_disallowed_broad_scan(self, command: str) -> bool:
        normalized = re.sub(r"\s+", " ", command).strip().lower()
        return (
            "find /" in normalized
            or "find ~" in normalized
            or "find $home" in normalized
        )

    def _elapsed_millis(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def _format_path_error(self, prefix: str, path: Path, exc: Exception) -> str:
        message = f"{prefix}: {exc}"
        if path.is_absolute() and not path.exists():
            message += (
                f"\n当前项目根目录: {self.base_dir}"
                "\n如果你是在读取当前项目文件，请优先使用相对路径，"
                "例如 README.md、pyproject.toml 或 .gitignore。"
            )
        return message
