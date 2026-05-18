import json
import tempfile
import unittest
from pathlib import Path
from queue import Queue

from paicli.agent.agent_message import AgentMessage, AgentMessageType
from paicli.agent.agent_orchestrator import AgentOrchestrator
from paicli.agent.agent_role import AgentRole
from paicli.agent.execution_step import ExecutionStep, StepStatus, StepType
from paicli.agent.sub_agent import PLANNER_PROMPT, SubAgent
from paicli.llm.glm_client import ChatResponse, FunctionCall, Message, ToolCall
from paicli.memory import LongTermMemory, MemoryManager
from paicli.tool.tool_registry import ToolRegistry


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
        self.context_queries = []
        self.tool_results = []
        self.extracted = False

    def build_context_for_query(self, query, max_tokens=500, limit=None):
        self.context_queries.append((query, max_tokens, limit))
        return "[相关记忆]\n- FACT: 用户喜欢 JDK 17"

    def add_tool_result(self, tool_name, result):
        self.tool_results.append((tool_name, result))
        super().add_tool_result(tool_name, result)

    def extract_and_save_facts(self):
        self.extracted = True
        return []


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def hybrid_search(self, query, top_k=5, project_path="", max_per_file=2):
        from paicli.rag import CodeChunk, SearchResult

        self.calls.append((query, top_k, project_path, max_per_file))
        return [
            SearchResult(
                CodeChunk.method_chunk("paicli/agent/agent.py", "Agent.run", "def run(self): pass", 1, 3),
                0.91,
            )
        ]


class MultiAgentTest(unittest.TestCase):
    def test_agent_role_exposes_display_name_and_description(self) -> None:
        self.assertEqual(AgentRole.PLANNER.display_name, "规划者")
        self.assertIn("拆解", AgentRole.PLANNER.description)
        self.assertEqual(AgentRole.WORKER.display_name, "执行者")
        self.assertEqual(AgentRole.REVIEWER.display_name, "检查者")

    def test_agent_message_carries_sender_role_content_and_type(self) -> None:
        message = AgentMessage(
            from_agent="planner",
            from_role=AgentRole.PLANNER,
            content="{}",
            type=AgentMessageType.TASK,
        )

        self.assertEqual(message.from_agent, "planner")
        self.assertEqual(message.from_role, AgentRole.PLANNER)
        self.assertEqual(message.type, AgentMessageType.TASK)

    def test_sub_agent_only_worker_uses_tools_and_clears_history(self) -> None:
        llm = FakeLLMClient(
            [
                ChatResponse(message=Message.assistant('{"summary":"ok","steps":[]}')),
                ChatResponse(
                    message=Message.assistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(
                                    name="write_file",
                                    arguments=json.dumps({"path": "hello.txt", "content": "hello"}),
                                ),
                            )
                        ],
                    )
                ),
                ChatResponse(message=Message.assistant("完成")),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(base_dir=tmp)
            planner = SubAgent("planner", AgentRole.PLANNER, llm, registry)
            worker = SubAgent("worker-1", AgentRole.WORKER, llm, registry)

            planner.execute(AgentMessage.user_task("写计划"))
            worker.execute(AgentMessage.user_task("写文件"))
            worker.clear_history()

            self.assertEqual(len(llm.calls[0][1]), 0)
            self.assertGreater(len(llm.calls[1][1]), 0)
            self.assertEqual((Path(tmp) / "hello.txt").read_text(encoding="utf-8"), "hello")
            self.assertEqual(len(worker.conversation_history), 1)
            self.assertEqual(worker.conversation_history[0].role, "system")

    def test_planner_prompt_guides_code_understanding_steps_to_search_code(self) -> None:
        self.assertIn("search_code", PLANNER_PROMPT)
        self.assertIn("代码理解", PLANNER_PROMPT)
        self.assertIn("定位实现", PLANNER_PROMPT)

    def test_worker_search_code_uses_shared_indexed_project_path(self) -> None:
        retriever = FakeRetriever()
        llm = FakeLLMClient(
            [
                ChatResponse(
                    message=Message.assistant(
                        "",
                        [
                            ToolCall(
                                id="call_search",
                                function=FunctionCall(
                                    name="search_code",
                                    arguments=json.dumps({"query": "Agent run", "top_k": "3"}),
                                ),
                            )
                        ],
                    )
                ),
                ChatResponse(message=Message.assistant("已根据检索结果定位 Agent.run")),
            ]
        )
        registry = ToolRegistry(code_retriever_factory=lambda: retriever)
        registry.set_project_path("/indexed/project")
        worker = SubAgent("worker-1", AgentRole.WORKER, llm, registry)

        result = worker.execute(AgentMessage.user_task("理解 Agent.run 的实现"))

        self.assertEqual(result.content, "已根据检索结果定位 Agent.run")
        self.assertEqual(retriever.calls, [("Agent run", 3, "/indexed/project", 2)])

    def test_orchestrator_parses_plan_and_filters_executable_steps(self) -> None:
        orchestrator = AgentOrchestrator("test-key", llm_client=FakeLLMClient([]))

        steps = orchestrator.parse_plan(
            json.dumps(
                {
                    "summary": "创建并验证",
                    "steps": [
                        {
                            "id": "step_1",
                            "description": "写入文件",
                            "type": "FILE_WRITE",
                            "dependencies": [],
                        },
                        {
                            "id": "step_2",
                            "description": "读取文件",
                            "type": "FILE_READ",
                            "dependencies": ["step_1"],
                        },
                    ],
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual([step.id for step in steps], ["step_1", "step_2"])
        self.assertEqual(orchestrator.get_executable_steps(steps), [steps[0]])
        steps[0].mark_completed("ok")
        self.assertEqual(orchestrator.get_executable_steps(steps), [steps[1]])

    def test_parse_review_approval_is_conservative(self) -> None:
        orchestrator = AgentOrchestrator("test-key", llm_client=FakeLLMClient([]))

        self.assertTrue(orchestrator.parse_review_approval('{"approved": true}'))
        self.assertFalse(orchestrator.parse_review_approval('{"approved": false}'))
        self.assertFalse(orchestrator.parse_review_approval("{}"))
        self.assertFalse(orchestrator.parse_review_approval(""))
        self.assertFalse(orchestrator.parse_review_approval("看起来还行"))
        self.assertFalse(orchestrator.parse_review_approval("未通过，需要修改"))
        self.assertTrue(orchestrator.parse_review_approval("通过，结果合格"))

    def test_run_retries_rejected_step_with_feedback_then_completes(self) -> None:
        plan = {
            "summary": "写文件并验收",
            "steps": [
                {
                    "id": "step_1",
                    "description": "写入 hello.txt",
                    "type": "FILE_WRITE",
                    "dependencies": [],
                }
            ],
        }
        llm = FakeLLMClient(
            [
                ChatResponse(message=Message.assistant(json.dumps(plan, ensure_ascii=False))),
                ChatResponse(
                    message=Message.assistant(
                        "",
                        [
                            ToolCall(
                                id="call_bad",
                                function=FunctionCall(
                                    name="write_file",
                                    arguments=json.dumps({"path": "hello.txt", "content": "bad"}),
                                ),
                            )
                        ],
                    )
                ),
                ChatResponse(message=Message.assistant("已写入 bad")),
                ChatResponse(
                    message=Message.assistant(
                        json.dumps(
                            {
                                "approved": False,
                                "summary": "内容不符合要求",
                                "issues": ["内容应该是 hello pai"],
                                "suggestions": ["重新写入正确内容"],
                            },
                            ensure_ascii=False,
                        )
                    )
                ),
                ChatResponse(
                    message=Message.assistant(
                        "",
                        [
                            ToolCall(
                                id="call_good",
                                function=FunctionCall(
                                    name="write_file",
                                    arguments=json.dumps({"path": "hello.txt", "content": "hello pai"}),
                                ),
                            )
                        ],
                    )
                ),
                ChatResponse(message=Message.assistant("已写入 hello pai")),
                ChatResponse(message=Message.assistant('{"approved": true, "summary": "通过"}')),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            memory = RecordingMemoryManager(tmp)
            orchestrator = AgentOrchestrator(
                "test-key",
                llm_client=llm,
                tool_registry=ToolRegistry(base_dir=tmp),
                memory_manager=memory,
            )

            result = orchestrator.run("写入 hello.txt，内容是 hello pai，每一步验收")

            self.assertEqual((Path(tmp) / "hello.txt").read_text(encoding="utf-8"), "hello pai")
            self.assertIn("✅ [step_1] 写入 hello.txt", result)
            self.assertTrue(memory.extracted)
            worker_retry_prompt = llm.calls[4][0][1].content
            self.assertIn("之前的执行结果被审查拒绝", worker_retry_prompt)
            self.assertIn("内容应该是 hello pai", worker_retry_prompt)

    def test_failed_step_skips_dependent_steps(self) -> None:
        steps = [
            ExecutionStep("step_1", "失败步骤", StepType.COMMAND),
            ExecutionStep("step_2", "依赖步骤", StepType.VERIFICATION, dependencies=["step_1"]),
        ]
        steps[0].mark_failed("执行失败")
        orchestrator = AgentOrchestrator("test-key", llm_client=FakeLLMClient([]))

        orchestrator.mark_blocked_steps_skipped(steps)

        self.assertEqual(steps[1].status, StepStatus.SKIPPED)
        self.assertIn("依赖步骤失败", steps[1].error)

    def test_parallel_batch_flushes_events_in_step_order(self) -> None:
        events = []
        orchestrator = AgentOrchestrator(
            "test-key",
            llm_client=FakeLLMClient([]),
            on_event=events.append,
        )
        steps = [
            ExecutionStep("step_1", "第一步", StepType.ANALYSIS),
            ExecutionStep("step_2", "第二步", StepType.ANALYSIS),
        ]
        worker_pool = Queue()
        for worker in orchestrator.workers:
            worker_pool.put(worker)

        def fake_run_step(step, all_steps, worker, emit=None):
            emit(f"{step.id}: start")
            emit(f"{step.id}: done")
            step.mark_completed("ok")

        orchestrator.run_step = fake_run_step

        orchestrator.run_batch_parallel(steps, steps, worker_pool)

        self.assertEqual(
            events,
            ["step_1: start", "step_1: done", "step_2: start", "step_2: done"],
        )


if __name__ == "__main__":
    unittest.main()
