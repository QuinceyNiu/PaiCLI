"""Role-specific lightweight agents for Multi-Agent mode."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from paicli.agent.browser_prompt import BROWSER_MCP_GUIDE
from paicli.agent.agent_message import AgentMessage, AgentMessageType
from paicli.agent.agent_role import AgentRole
from paicli.llm.glm_client import ChatResponse, Message, ToolCall
from paicli.skill import SkillContextBuffer, SkillRegistry, active_skill_context
from paicli.tool.tool_registry import ToolInvocation, ToolRegistry


LOGGER = logging.getLogger(__name__)
MAX_ITERATIONS = 8

PLANNER_PROMPT = """
你是一个任务规划专家。你的职责是分析用户的需求，将其拆解为清晰的执行步骤。

请按以下 JSON 格式输出执行计划：
{
    "summary": "任务摘要",
    "steps": [
        {
            "id": "step_1",
            "description": "步骤描述，要具体明确",
            "type": "FILE_READ | FILE_WRITE | COMMAND | ANALYSIS | VERIFICATION",
            "dependencies": []
        }
    ]
}

规划规则：
1. 涉及代码理解、定位实现、调用关系、重构影响分析或查找相关代码的任务，应在步骤描述中明确要求先使用 search_code 检索代码库上下文。
2. search_code 只是执行者可调用的工具，规划者不要自己调用工具。

简单任务拆 1-3 步，复杂任务拆 5-10 步。只输出 JSON，不要输出 Markdown。
""".strip()

WORKER_PROMPT = """
你是一个任务执行专家。你的职责是根据给定的任务步骤，调用工具完成具体操作。

可用工具：
1. read_file - 读取文件内容
2. write_file - 写入文件内容
3. list_dir - 列出目录内容
4. execute_command - 执行命令
5. create_project - 创建项目
6. search_code - 语义检索代码库（如果可用）

如果任务涉及理解代码库，请优先使用 search_code 工具；如果 search_code 不可用，再使用 read_file、list_dir 等工具。
同一轮返回多个工具调用时，系统会并行执行这些工具；如果工具之间有依赖关系，请分多轮调用。
请用中文简洁汇报执行结果。
""".strip() + BROWSER_MCP_GUIDE

REVIEWER_PROMPT = """
你是一个质量检查专家。你的职责是检查执行结果是否正确、完整和高质量。

请以 JSON 格式输出检查结果：
{
    "approved": true 或 false,
    "summary": "检查摘要",
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
}

只输出 JSON，不要输出 Markdown。
""".strip()


class ChatClient(Protocol):
    def chat(self, messages: list[Message], tools: list[Any]) -> ChatResponse:
        """Call the LLM with conversation history and optional tools."""


class SubAgent:
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_client: ChatClient,
        tool_registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
        skill_registry: SkillRegistry | None = None,
        skill_context_buffer: SkillContextBuffer | None = None,
        loaded_skill_names: set[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.skill_registry = skill_registry
        self.skill_context_buffer = skill_context_buffer or SkillContextBuffer()
        self.loaded_skill_names = loaded_skill_names if loaded_skill_names is not None else set()
        if self.skill_registry is not None and self.should_use_tools():
            for tool in self.skill_registry.tool_provider(
                self.skill_context_buffer,
                self.loaded_skill_names,
            ).get_tools():
                self.tool_registry.register(tool)
        self.system_prompt = self._build_system_prompt(role)
        self.conversation_history: list[Message] = [Message.system(self.system_prompt)]

    def execute(self, message: AgentMessage) -> AgentMessage:
        self.conversation_history.append(Message.user(self._user_message_with_skill_context(message.content)))
        for _ in range(self.max_iterations):
            try:
                response = self.llm_client.chat(
                    self.conversation_history,
                    self.tool_registry.tools_for_llm() if self.should_use_tools() else [],
                )
            except Exception as exc:
                LOGGER.exception("SubAgent chat failed")
                return AgentMessage(self.agent_id, self.role, f"执行错误: {exc}", AgentMessageType.ERROR)

            assistant_message = response.message
            if assistant_message.tool_calls and self.should_use_tools():
                self.conversation_history.append(
                    Message.assistant(assistant_message.content or "", assistant_message.tool_calls)
                )
                tool_results = self._execute_tool_calls(assistant_message.tool_calls)
                for tool_result in tool_results:
                    self.conversation_history.append(Message.tool(tool_result.id, tool_result.result))
                skill_context = self.skill_context_buffer.drain()
                if skill_context:
                    self.conversation_history.append(
                        Message.user(self._format_loaded_skill_user_message(skill_context, message.content))
                    )
                continue

            content = assistant_message.content or ""
            self.conversation_history.append(Message.assistant(content))
            return AgentMessage(self.agent_id, self.role, content, AgentMessageType.RESULT)

        return AgentMessage(self.agent_id, self.role, "达到最大迭代次数限制", AgentMessageType.ERROR)

    def review(self, task_description: str, execution_result: str) -> AgentMessage:
        content = "\n".join(
            [
                f"原始任务：{task_description}",
                "",
                "执行结果：",
                execution_result,
            ]
        )
        return self.execute(AgentMessage("orchestrator", AgentRole.REVIEWER, content, AgentMessageType.TASK))

    def clear_history(self) -> None:
        self.skill_context_buffer.clear()
        self.loaded_skill_names.clear()
        self.conversation_history = [Message.system(self.system_prompt)]

    def should_use_tools(self) -> bool:
        return self.role == AgentRole.WORKER

    def _execute_tool_calls(self, tool_calls: list[ToolCall]):
        invocations = [
            ToolInvocation(tool_call.id, tool_call.function.name, tool_call.function.arguments or "{}")
            for tool_call in tool_calls
        ]
        if self.skill_registry is not None and any(invocation.name == "load_skill" for invocation in invocations):
            with active_skill_context(self.skill_context_buffer, self.loaded_skill_names):
                return [
                    self.tool_registry.execute_tools([invocation])[0]
                    for invocation in invocations
                ]
        return self.tool_registry.execute_tools(invocations)

    def _build_system_prompt(self, role: AgentRole) -> str:
        prompt = self._prompt_for_role(role)
        if self.skill_registry is None or not self.should_use_tools():
            return prompt
        index = self.skill_registry.format_index()
        if not index:
            return prompt
        return f"{prompt}\n\n{index}"

    def _user_message_with_skill_context(self, content: str) -> str:
        skill_context = self.skill_context_buffer.drain()
        if not skill_context:
            return content
        return self._format_loaded_skill_user_message(skill_context, content)

    def _format_loaded_skill_user_message(self, skill_context: str, content: str) -> str:
        return f"{skill_context}\n\n---\n用户输入：{content}"

    def _prompt_for_role(self, role: AgentRole) -> str:
        if role == AgentRole.PLANNER:
            return PLANNER_PROMPT
        if role == AgentRole.WORKER:
            return WORKER_PROMPT
        return REVIEWER_PROMPT
