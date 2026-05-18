import json
import tempfile
import unittest
from pathlib import Path

from paicli.agent.agent import Agent
from paicli.llm.glm_client import ChatResponse, FunctionCall, Message, ToolCall, Usage
from paicli.memory import LongTermMemory, MemoryManager, MemoryType


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return self.responses.pop(0)


class RecordingMemoryManager(MemoryManager):
    def __init__(self, storage_dir):
        super().__init__(long_term_memory=LongTermMemory(storage_dir=storage_dir))
        self.context_limits = []

    def build_context_for_query(
        self,
        query: str,
        max_tokens: int = 500,
        limit: int | None = None,
    ) -> str:
        self.context_limits.append(max_tokens)
        return super().build_context_for_query(query, max_tokens=max_tokens, limit=limit)


class AgentTest(unittest.TestCase):
    def test_run_executes_tool_calls_until_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLMClient(
                [
                    ChatResponse(
                        message=Message.assistant(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    function=FunctionCall(
                                        name="write_file",
                                        arguments=json.dumps({"path": "hello.txt", "content": "hi"}),
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    ),
                    ChatResponse(message=Message.assistant("已写入 hello.txt")),
                ]
            )
            agent = Agent(api_key="test-key", llm_client=llm, base_dir=tmp)

            result = agent.run("写一个 hello.txt")

            self.assertEqual(result, "已写入 hello.txt")
            self.assertEqual((Path(tmp) / "hello.txt").read_text(), "hi")
            self.assertEqual(len(llm.calls), 2)
            self.assertEqual(agent.conversation_history[-2].role, "tool")
            self.assertIn("文件已写入", agent.conversation_history[-2].content)


    def test_run_emits_process_events(self) -> None:
        events = []
        llm = FakeLLMClient(
            [
                ChatResponse(
                    message=Message.assistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(
                                    name="write_file",
                                    arguments=json.dumps({"path": "hello.txt", "content": "hi"}),
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
                ChatResponse(message=Message.assistant("完成")),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(api_key="test-key", llm_client=llm, base_dir=tmp, on_event=events.append)

            agent.run("写文件")

        self.assertEqual(events[0].type, "thinking")
        self.assertEqual(events[1].type, "tool")
        self.assertEqual(events[1].tool_name, "write_file")
        self.assertEqual(events[1].arguments["path"], "hello.txt")
        self.assertIn("文件已写入", events[1].result)
        self.assertEqual(events[2].type, "thinking")

    def test_run_returns_iteration_limit_message(self) -> None:
        repeated = ChatResponse(
            message=Message.assistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_loop",
                        function=FunctionCall(name="list_dir", arguments=json.dumps({"path": "."})),
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
        llm = FakeLLMClient([repeated] * 10)
        agent = Agent(api_key="test-key", llm_client=llm, max_iterations=10)

        result = agent.run("一直列目录")

        self.assertEqual(result, "达到最大迭代次数限制")

    def test_clear_history_keeps_system_prompt(self) -> None:
        agent = Agent(api_key="test-key", llm_client=FakeLLMClient([]))
        agent.conversation_history.append(Message.user("hello"))

        agent.clear_history()

        self.assertEqual(len(agent.conversation_history), 1)
        self.assertEqual(agent.conversation_history[0].role, "system")

    def test_invalid_tool_arguments_are_reported_to_model(self) -> None:
        llm = FakeLLMClient(
            [
                ChatResponse(
                    message=Message.assistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="bad_args",
                                function=FunctionCall(name="read_file", arguments="not json"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                ),
                ChatResponse(message=Message.assistant("参数格式不对")),
            ]
        )
        agent = Agent(api_key="test-key", llm_client=llm)

        result = agent.run("读取文件")

        self.assertEqual(result, "参数格式不对")
        self.assertIn("工具参数解析失败", agent.conversation_history[-2].content)

    def test_run_injects_relevant_memory_context_into_llm_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLMClient([ChatResponse(message=Message.assistant("用 JDK 17"))])
            memory_manager = RecordingMemoryManager(tmp)
            memory_manager.store_message("用户喜欢 JDK 17", MemoryType.FACT)
            agent = Agent(
                api_key="test-key",
                llm_client=llm,
                memory_manager=memory_manager,
            )

            agent.run("按 JDK 生成 Java 项目")

        first_call_messages = llm.calls[0][0]
        self.assertEqual(first_call_messages[0].role, "system")
        self.assertIn("[相关记忆]", first_call_messages[0].content)
        self.assertIn("用户喜欢 JDK 17", first_call_messages[0].content)
        self.assertEqual(first_call_messages[1].role, "user")
        self.assertEqual(first_call_messages[1].content, "按 JDK 生成 Java 项目")
        self.assertEqual(memory_manager.context_limits, [500])

    def test_clear_history_extracts_facts_then_clears_short_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLMClient([ChatResponse(message=Message.assistant("- 项目使用 Maven 构建"))])
            memory_manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))
            agent = Agent(api_key="test-key", llm_client=llm, memory_manager=memory_manager)
            memory_manager.add_user_message("项目使用 Maven 构建")

            facts = agent.clear_history()

            self.assertEqual(facts, ["项目使用 Maven 构建"])
            self.assertEqual(agent.conversation_history, [Message.system(agent.system_prompt)])
            self.assertEqual(memory_manager.conversation_memory.entries(), [])

    def test_run_records_token_usage_in_memory_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))
            llm = FakeLLMClient(
                [
                    ChatResponse(
                        message=Message.assistant("完成"),
                        usage=Usage(prompt_tokens=123, completion_tokens=45, total_tokens=168),
                    )
                ]
            )
            agent = Agent(api_key="test-key", llm_client=llm, memory_manager=memory_manager)

            agent.run("你好")

            self.assertEqual(memory_manager.token_budget.llm_call_count, 1)
            self.assertEqual(memory_manager.token_budget.total_input_tokens, 123)
            self.assertEqual(memory_manager.token_budget.total_output_tokens, 45)

    def test_default_memory_compressor_uses_agent_llm_client(self) -> None:
        llm = FakeLLMClient([ChatResponse(message=Message.assistant("完成"))])

        agent = Agent(api_key="test-key", llm_client=llm)

        self.assertIs(agent.memory_manager.context_compressor.llm_client, llm)

    def test_system_prompt_guides_model_to_search_code_for_codebase_questions(self) -> None:
        agent = Agent(api_key="test-key", llm_client=FakeLLMClient([]))

        self.assertIn("search_code", agent.system_prompt)
        self.assertIn("代码库", agent.system_prompt)


if __name__ == "__main__":
    unittest.main()
