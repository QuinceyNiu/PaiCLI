"""PaiCli Agent core: a small ReAct loop over GLM tool calls."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from paicli.llm.glm_client import ChatResponse, GLMClient, Message, ToolCall
from paicli.memory import MemoryManager
from paicli.tool.tool_registry import ToolRegistry


LOGGER = logging.getLogger(__name__)
MAX_ITERATIONS = 10
SYSTEM_PROMPT = """
你是一个智能编程助手，可以帮助用户完成各种任务。

你可以使用以下工具来完成任务：
1. read_file - 读取文件内容
2. write_file - 写入文件内容
3. list_dir - 列出目录内容
4. execute_command - 执行Shell命令
5. create_project - 创建新项目结构
6. search_code - 语义检索代码库，查找相关类、方法和代码片段

当需要操作文件、执行命令或创建项目时，请使用工具调用。
如果用户询问与代码库相关的问题（例如“这个类是干什么的”、“某个方法怎么实现”、“哪里用了某个功能”），请优先使用 search_code 检索相关代码，再基于检索结果回答。
使用工具后，根据工具返回的结果继续思考下一步行动。

请用中文回复用户。
""".strip()


@dataclass(frozen=True)
class AgentEvent:
    type: str
    tool_name: Optional[str] = None
    arguments: dict[str, str] = field(default_factory=dict)
    result: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


AgentEventHandler = Callable[[AgentEvent], None]


class ChatClient(Protocol):
    def chat(self, messages: list[Message], tools: list[Any]) -> ChatResponse:
        """Call the LLM with conversation history and tool definitions."""


class Agent:
    def __init__(
        self,
        api_key: str,
        llm_client: Optional[ChatClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        base_dir: str | Path | None = None,
        max_iterations: int = MAX_ITERATIONS,
        on_event: Optional[AgentEventHandler] = None,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.llm_client = llm_client or GLMClient(api_key)
        self.tool_registry = tool_registry or ToolRegistry(base_dir=base_dir)
        self.max_iterations = max_iterations
        self.on_event = on_event
        self.memory_manager = memory_manager or MemoryManager()
        if self.memory_manager.context_compressor.llm_client is None:
            self.memory_manager.context_compressor.llm_client = self.llm_client
        self.system_prompt = SYSTEM_PROMPT
        self.conversation_history: list[Message] = [Message.system(SYSTEM_PROMPT)]

    def clear_history(self) -> list[str]:
        facts = self.memory_manager.extract_and_save_facts()
        self.memory_manager.clear_short_term()
        self.conversation_history = [Message.system(self.system_prompt)]
        return facts

    def run(self, user_input: str) -> str:
        self.memory_manager.add_user_message(user_input)
        memory_context = self.memory_manager.build_context_for_query(user_input, max_tokens=500)
        self._update_system_prompt_with_memory(memory_context)
        self.conversation_history.append(Message.user(user_input))

        for _ in range(self.max_iterations):
            self._emit(AgentEvent(type="thinking"))
            try:
                response = self.llm_client.chat(
                    self.conversation_history,
                    self.tool_registry.tools_for_llm(),
                )
            except Exception as exc:
                LOGGER.exception("LLM chat failed")
                return f"执行错误: {exc}"

            self._emit_usage(response)

            message = response.message
            if message.tool_calls:
                self.conversation_history.append(
                    Message.assistant(message.content or "", message.tool_calls)
                )
                for tool_call in message.tool_calls:
                    result, args = self._execute_tool_call(tool_call)
                    self._emit(
                        AgentEvent(
                            type="tool",
                            tool_name=tool_call.function.name,
                            arguments=args,
                            result=result,
                        )
                    )
                    self.conversation_history.append(Message.tool(tool_call.id, result))
                    self.memory_manager.add_tool_result(tool_call.function.name, result)
                continue

            content = message.content or ""
            self.conversation_history.append(Message.assistant(content))
            self.memory_manager.add_assistant_message(content)
            return content

        return "达到最大迭代次数限制"

    def _update_system_prompt_with_memory(self, memory_context: str) -> None:
        if not memory_context:
            self.conversation_history[0] = Message.system(self.system_prompt)
            return
        self.conversation_history[0] = Message.system(
            f"{self.system_prompt}\n\n{memory_context}"
        )

    def _emit(self, event: AgentEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _emit_usage(self, response: ChatResponse) -> None:
        usage = response.usage
        if usage.total_tokens <= 0 and usage.prompt_tokens <= 0 and usage.completion_tokens <= 0:
            return
        self.memory_manager.record_token_usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )
        self._emit(
            AgentEvent(
                type="usage",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        )

    def _execute_tool_call(self, tool_call: ToolCall) -> tuple[str, dict[str, str]]:
        try:
            raw_args = tool_call.function.arguments or "{}"
            parsed_args = json.loads(raw_args)
            if not isinstance(parsed_args, dict):
                return "工具参数解析失败: arguments 必须是 JSON 对象", {}
            args = {
                str(key): value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in parsed_args.items()
            }
        except Exception as exc:
            return f"工具参数解析失败: {exc}", {}

        return self.tool_registry.execute(tool_call.function.name, args), args
