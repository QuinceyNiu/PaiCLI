import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from paicli.agent.agent import AgentEvent
from paicli.agent.agent import SYSTEM_PROMPT
from paicli.cli.main import (
    BANNER,
    clear_agent_history,
    finalize_memory_session,
    format_agent_event,
    format_graph_report,
    format_memory_report,
    format_plan_summary,
    format_startup_status,
    format_mcp_status,
    handle_team_interaction,
    handle_plan_interaction,
    handle_hitl_command,
    index_codebase,
    load_api_key,
    main,
    search_codebase,
    save_memory_fact,
)
from paicli.memory import ConversationMemory, LongTermMemory, MemoryManager, TokenBudget
from paicli.plan.execution_plan import ExecutionPlan, PlanStatus
from paicli.plan.task import Task, TaskStatus, TaskType
from paicli.rag import CodeChunk, CodeRelation, SearchResult


class FakePlanAgent:
    def __init__(self):
        self.created_goals = []
        self.executed_plans = []

    def create_plan(self, goal):
        self.created_goals.append(goal)
        plan = ExecutionPlan(f"plan_{len(self.created_goals)}", goal)
        task = Task("task_1", "写入文件", TaskType.FILE_WRITE)
        plan.add_task(task)
        plan.compute_execution_order()
        return plan

    def execute_plan(self, plan):
        self.executed_plans.append(plan)
        plan.tasks["task_1"].mark_started()
        plan.tasks["task_1"].mark_completed("ok")
        plan.mark_completed("全部任务执行完成")
        return "执行完成"


class FakeTeamAgent:
    def __init__(self):
        self.inputs = []

    def run(self, user_input):
        self.inputs.append(user_input)
        return "团队协作完成"


class FakeGraphStore:
    def __init__(self, relations):
        self.relations = relations
        self.calls = []

    def find_relations(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs.get("source") or kwargs.get("target")
        if name is None:
            return list(self.relations)
        return [
            relation for relation in self.relations
            if relation.source == name or relation.target == name
        ]


class FakeToolRegistry:
    instances = []

    def __init__(self, *args, **kwargs):
        self.project_paths = []
        FakeToolRegistry.instances.append(self)

    def set_project_path(self, project_path):
        self.project_paths.append(project_path)


class FakeHitlHandler:
    instances = []

    def __init__(self, enabled=False, *args, **kwargs):
        self.enabled = enabled
        self.cleared = False
        FakeHitlHandler.instances.append(self)

    def is_enabled(self):
        return self.enabled

    def set_enabled(self, enabled):
        self.enabled = enabled

    def clear_approved_all(self):
        self.cleared = True


class CliTest(unittest.TestCase):
    def test_banner_mentions_memory_enhanced_version(self) -> None:
        self.assertIn("Memory-Enhanced Agent CLI v3.0.0", BANNER)

    def test_format_startup_status_reports_api_key_and_default_mode(self) -> None:
        status = format_startup_status(api_key_loaded=True, plan_mode=False)

        self.assertIn("✅ API Key 已加载", status)
        self.assertIn("🔄 使用 ReAct 模式", status)

    def test_format_mcp_status_reports_configured_and_missing_servers(self) -> None:
        self.assertEqual(format_mcp_status([]), "🔌 MCP server：未配置")

        status = format_mcp_status(["filesystem", "zread"])

        self.assertEqual(status, "🔌 MCP server：filesystem, zread")

    def test_system_prompt_mentions_web_tools(self) -> None:
        self.assertIn("web_search", SYSTEM_PROMPT)
        self.assertIn("web_fetch", SYSTEM_PROMPT)
        self.assertIn("MCP", SYSTEM_PROMPT)

    def test_load_api_key_prefers_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("GLM_API_KEY=from-file\n", encoding="utf-8")

            with patch.dict(os.environ, {"GLM_API_KEY": "from-env"}, clear=False):
                self.assertEqual(load_api_key(env_file), "from-file")

    def test_load_api_key_falls_back_to_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".env"

            with patch.dict(os.environ, {"GLM_API_KEY": "from-env"}, clear=False):
                self.assertEqual(load_api_key(missing), "from-env")

    def test_load_api_key_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".env"

            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(load_api_key(missing))

    def test_format_tool_event_matches_verbose_style(self) -> None:
        event = AgentEvent(
            type="tool",
            tool_name="write_file",
            arguments={"path": "hello.txt", "content": "hello pai"},
            result="文件已写入: hello.txt",
        )

        output = format_agent_event(event)

        self.assertIn("🔧 执行工具: write_file", output)
        self.assertIn('"path": "hello.txt"', output)
        self.assertIn("结果: 文件已写入: hello.txt", output)

    def test_format_usage_event(self) -> None:
        event = AgentEvent(type="usage", prompt_tokens=312, completion_tokens=156)

        self.assertEqual(format_agent_event(event), "📊 Token使用: 输入=312, 输出=156")

    def test_format_memory_report_shows_memory_and_token_budget_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(
                conversation_memory=ConversationMemory(max_tokens=100),
                long_term_memory=LongTermMemory(storage_dir=tmp),
                token_budget=TokenBudget(
                    context_window=100,
                    reserved_for_system=10,
                    reserved_for_tools=0,
                    reserved_for_response=0,
                ),
            )
            manager.token_budget.record_usage(input_tokens=312, output_tokens=156)

            report = format_memory_report(manager)

            self.assertIn("记忆系统状态:", report)
            self.assertIn("短期记忆: 0条 / 0 tokens", report)
            self.assertIn("长期记忆: 0条 / 0 tokens", report)
            self.assertIn("Token 统计: 调用 1 次 | 总输入: 312 | 总输出: 156", report)

    def test_format_graph_report_renders_relations_for_named_symbol(self) -> None:
        store = FakeGraphStore(
            [
                CodeRelation("Agent", "Agent.run", "contains", "agent.py"),
                CodeRelation("Agent", "BaseAgent", "extends", "agent.py"),
                CodeRelation("Agent.run", "chat", "calls", "agent.py"),
            ]
        )

        report = format_graph_report("Agent", store, project_path="/repo")

        self.assertIn("🕸️ 查询类关系图谱: Agent", report)
        self.assertIn("📋 找到 3 条关系:", report)
        self.assertIn("Agent ├── contains --> Agent.run", report)
        self.assertIn("Agent └── extends --> BaseAgent", report)
        self.assertIn("Agent.run └── calls --> chat", report)

    def test_index_codebase_indexes_path_and_syncs_tool_registry_project_path(self) -> None:
        class FakeCodeIndex:
            def __init__(self, vector_store, embedding_client=None, on_progress=None, on_warning=None):
                self.vector_store = vector_store
                self.embedding_client = embedding_client

            def index_with_stats(self, project_root):
                from paicli.rag import IndexStats

                return IndexStats(str(Path(project_root).resolve()), 2, 7, 4)

        with tempfile.TemporaryDirectory() as tmp:
            registry = FakeToolRegistry()
            with patch("paicli.cli.main.CodeIndex", FakeCodeIndex):
                report = index_codebase(tmp, object(), registry)

            self.assertIn("发现 2 个文件待索引", report)
            self.assertIn("索引完成: 7 个代码块, 4 条关系", report)
            self.assertEqual(registry.project_paths, [str(Path(tmp).resolve())])

    def test_search_codebase_requires_index_before_searching(self) -> None:
        class EmptyStore:
            def all_chunks(self, project_path):
                return []

        result = search_codebase("Agent run", EmptyStore(), project_path="/repo")

        self.assertIn("代码库尚未索引，请先使用 /index 命令", result)

    def test_search_codebase_formats_cli_results_and_catches_errors(self) -> None:
        class IndexedStore:
            def all_chunks(self, project_path):
                return [CodeChunk.file_chunk("agent.py", "agent.py#1", "content", 1, 1)]

        class FakeRetriever:
            def __init__(self, vector_store):
                pass

            def hybrid_search(self, query, top_k, project_path):
                return [
                    SearchResult(
                        CodeChunk.method_chunk("agent.py", "Agent.run", "def run(self): pass", 10, 10),
                        1.0,
                    )
                ]

        with patch("paicli.cli.main.CodeRetriever", FakeRetriever):
            result = search_codebase("Agent run", IndexedStore(), project_path="/repo")

        self.assertIn("🔎 检索: Agent run", result)
        self.assertIn("找到 1 个相关代码块", result)
        self.assertIn("Agent.run", result)

    def test_save_memory_fact_persists_manual_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(long_term_memory=LongTermMemory(storage_dir=tmp))

            message = save_memory_fact(manager, "用户喜欢 JDK 17")

            self.assertEqual(message, "💾 已保存事实: 用户喜欢 JDK 17")
            self.assertEqual(manager.long_term_memory.entries()[0].content, "用户喜欢 JDK 17")

    def test_finalize_memory_session_extracts_facts_on_exit(self) -> None:
        class FakeMemoryManager:
            def __init__(self):
                self.extracted = False

            def extract_and_save_facts(self):
                self.extracted = True
                return ["用户喜欢 JDK 17"]

        manager = FakeMemoryManager()

        finalize_memory_session(manager)

        self.assertTrue(manager.extracted)

    def test_clear_agent_history_reports_extracted_fact_count(self) -> None:
        class FakeAgent:
            def clear_history(self):
                return ["用户喜欢 JDK 17", "项目使用 Python 3.12"]

        output = clear_agent_history(FakeAgent())

        self.assertIn("🧠 提取关键事实到长期记忆...", output)
        self.assertIn("提取了 2 条事实", output)
        self.assertIn("🗑️ 对话历史已清空，关键事实已保存到长期记忆", output)

    def test_handle_hitl_command_reports_and_toggles_status(self) -> None:
        handler = FakeHitlHandler(enabled=False)

        self.assertIn("关闭", handle_hitl_command("", handler))
        self.assertIn("已启用", handle_hitl_command("on", handler))
        self.assertTrue(handler.is_enabled())
        self.assertIn("开启", handle_hitl_command("", handler))
        self.assertIn("已关闭", handle_hitl_command("off", handler))
        self.assertFalse(handler.is_enabled())
        self.assertIn("用法", handle_hitl_command("bad", handler))

    def test_handle_plan_interaction_confirms_before_execution(self) -> None:
        agent = FakePlanAgent()
        output = StringIO()
        answers = iter(["y"])

        handle_plan_interaction(
            "创建 Java 项目，然后写 Hello 类，接着编译运行",
            agent,
            input_func=lambda _: next(answers),
            output=output,
        )

        self.assertEqual(agent.created_goals, ["创建 Java 项目，然后写 Hello 类，接着编译运行"])
        self.assertEqual(len(agent.executed_plans), 1)
        self.assertIn("执行计划", output.getvalue())
        self.assertIn("计划摘要", output.getvalue())
        self.assertIn("执行完成", output.getvalue())

    def test_format_plan_summary_shows_parallel_first_batch(self) -> None:
        plan = ExecutionPlan("plan_parallel", "读取多个文件后汇总")
        first = Task("task_1", "读取 pyproject.toml", TaskType.FILE_READ)
        second = Task("task_2", "读取 README.md", TaskType.FILE_READ)
        final = Task("task_3", "汇总", TaskType.ANALYSIS, dependencies=["task_1", "task_2"])
        plan.add_task(first)
        plan.add_task(second)
        plan.add_task(final)
        plan.compute_execution_order()

        summary = format_plan_summary(plan)

        self.assertIn("任务数: 3", summary)
        self.assertIn("并行批次: 2", summary)
        self.assertIn("当前可执行: 2", summary)
        self.assertIn("首批执行: task_1, task_2", summary)

    def test_handle_plan_interaction_replans_when_user_adds_context(self) -> None:
        agent = FakePlanAgent()
        output = StringIO()
        answers = iter(["请使用 Maven", "y"])

        handle_plan_interaction(
            "创建 Java 项目，然后写 Hello 类，接着编译运行",
            agent,
            input_func=lambda _: next(answers),
            output=output,
        )

        self.assertEqual(agent.created_goals[0], "创建 Java 项目，然后写 Hello 类，接着编译运行")
        self.assertEqual(
            agent.created_goals[1],
            "创建 Java 项目，然后写 Hello 类，接着编译运行\n补充要求：请使用 Maven",
        )
        self.assertEqual(len(agent.executed_plans), 1)

    def test_handle_plan_interaction_can_cancel_plan(self) -> None:
        agent = FakePlanAgent()
        output = StringIO()
        answers = iter(["n"])

        plan = handle_plan_interaction(
            "创建 Java 项目，然后写 Hello 类，接着编译运行",
            agent,
            input_func=lambda _: next(answers),
            output=output,
        )

        self.assertEqual(plan.status, PlanStatus.CANCELLED)
        self.assertEqual(plan.tasks["task_1"].status, TaskStatus.PENDING)
        self.assertEqual(agent.executed_plans, [])
        self.assertIn("已取消执行计划", output.getvalue())

    def test_handle_team_interaction_runs_orchestrator_once(self) -> None:
        agent = FakeTeamAgent()
        output = StringIO()

        handle_team_interaction("创建项目、写代码、跑测试，每一步验收", agent, output=output)

        self.assertEqual(agent.inputs, ["创建项目、写代码、跑测试，每一步验收"])
        self.assertIn("团队协作完成", output.getvalue())

    def test_main_team_mode_runs_next_task_then_returns_to_react(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                self.inputs = []

            def run(self, user_input):
                self.inputs.append(user_input)
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            instances = []

            def __init__(self, *args, **kwargs):
                self.inputs = []
                FakeAgentOrchestrator.instances.append(self)

            def run(self, user_input):
                self.inputs.append(user_input)
                return "team 完成"

        inputs = iter(["/team", "复杂任务", "普通问题", "exit"])
        output = StringIO()

        with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
            patch("paicli.cli.main.Agent", FakeAgent), \
            patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
            patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
            patch("builtins.input", lambda _: next(inputs)), \
            patch("sys.stdout", output):
            self.assertEqual(main([]), 0)

        self.assertEqual(FakeAgentOrchestrator.instances[0].inputs, ["复杂任务"])
        rendered = output.getvalue()
        self.assertIn("🤝 已进入 Multi-Agent 团队模式", rendered)
        self.assertIn("🤝 Team: team 完成", rendered)
        self.assertIn("🤖 Agent: react 完成", rendered)

    def test_main_inline_plan_command_runs_plan_interaction_immediately(self) -> None:
        class FakeAgent:
            instances = []

            def __init__(self, *args, **kwargs):
                self.inputs = []
                FakeAgent.instances.append(self)

            def run(self, user_input):
                self.inputs.append(user_input)
                return "react 完成"

        class FakePlanExecuteAgent(FakePlanAgent):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__()
                FakePlanExecuteAgent.instances.append(self)

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        inputs = iter(["/plan 分析 PaiCLI 项目", "y", "exit"])
        output = StringIO()

        with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
            patch("paicli.cli.main.Agent", FakeAgent), \
            patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
            patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
            patch("builtins.input", lambda _: next(inputs)), \
            patch("sys.stdout", output):
            self.assertEqual(main([]), 0)

        self.assertEqual(FakePlanExecuteAgent.instances[0].created_goals, ["分析 PaiCLI 项目"])
        self.assertEqual(len(FakePlanExecuteAgent.instances[0].executed_plans), 1)
        self.assertEqual(FakeAgent.instances[0].inputs, [])
        self.assertIn("🧭 使用 Plan-and-Execute 模式", output.getvalue())

    def test_main_graph_command_prints_relationship_report(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, user_input):
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        class FakeVectorStore:
            @classmethod
            def default(cls):
                return cls()

            def find_relations(self, **kwargs):
                return [CodeRelation("Agent", "Agent.run", "contains", "agent.py")]

        inputs = iter(["/graph Agent", "exit"])
        output = StringIO()

        with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
            patch("paicli.cli.main.Agent", FakeAgent), \
            patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
            patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
            patch("paicli.cli.main.VectorStore", FakeVectorStore), \
            patch("builtins.input", lambda _: next(inputs)), \
            patch("sys.stdout", output):
            self.assertEqual(main([]), 0)

        self.assertIn("Agent", output.getvalue())
        self.assertIn("└── contains --> Agent.run", output.getvalue())

    def test_main_index_command_indexes_and_syncs_registry(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, user_input):
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        class FakeVectorStore:
            @classmethod
            def default(cls):
                return cls()

        class FakeCodeIndex:
            def __init__(self, vector_store, embedding_client=None, on_progress=None, on_warning=None):
                pass

            def index_with_stats(self, project_root):
                from paicli.rag import IndexStats

                return IndexStats(str(Path(project_root).resolve()), 1, 3, 2)

        FakeToolRegistry.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            inputs = iter([f"/index {tmp}", "exit"])
            output = StringIO()

            with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
                patch("paicli.cli.main.Agent", FakeAgent), \
                patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
                patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
                patch("paicli.cli.main.HitlToolRegistry", FakeToolRegistry), \
                patch("paicli.cli.main.VectorStore", FakeVectorStore), \
                patch("paicli.cli.main.CodeIndex", FakeCodeIndex), \
                patch("builtins.input", lambda _: next(inputs)), \
                patch("sys.stdout", output):
                self.assertEqual(main([]), 0)

            self.assertEqual(FakeToolRegistry.instances[0].project_paths, [str(Path(tmp).resolve())])
            self.assertIn("索引完成: 3 个代码块, 2 条关系", output.getvalue())

    def test_main_hitl_command_controls_handler(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, user_input):
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        FakeHitlHandler.instances = []
        inputs = iter(["/hitl", "/hitl on", "/hitl", "/hitl off", "exit"])
        output = StringIO()

        with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
            patch("paicli.cli.main.Agent", FakeAgent), \
            patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
            patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
            patch("paicli.cli.main.TerminalHitlHandler", FakeHitlHandler), \
            patch("paicli.cli.main.HitlToolRegistry", FakeToolRegistry), \
            patch("builtins.input", lambda _: next(inputs)), \
            patch("sys.stdout", output):
            self.assertEqual(main([]), 0)

        rendered = output.getvalue()
        self.assertIn("HITL 当前状态: 关闭", rendered)
        self.assertIn("HITL 已启用", rendered)
        self.assertIn("HITL 当前状态: 开启", rendered)
        self.assertIn("HITL 已关闭", rendered)

    def test_main_clear_command_clears_hitl_approved_all(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def clear_history(self):
                return []

            def run(self, user_input):
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        FakeHitlHandler.instances = []
        inputs = iter(["/clear", "exit"])
        output = StringIO()

        with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
            patch("paicli.cli.main.Agent", FakeAgent), \
            patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
            patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
            patch("paicli.cli.main.TerminalHitlHandler", FakeHitlHandler), \
            patch("paicli.cli.main.HitlToolRegistry", FakeToolRegistry), \
            patch("builtins.input", lambda _: next(inputs)), \
            patch("sys.stdout", output):
            self.assertEqual(main([]), 0)

        self.assertTrue(FakeHitlHandler.instances[0].cleared)

    def test_main_search_command_prints_search_results(self) -> None:
        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, user_input):
                return "react 完成"

        class FakePlanExecuteAgent:
            def __init__(self, *args, **kwargs):
                pass

        class FakeAgentOrchestrator:
            def __init__(self, *args, **kwargs):
                pass

        class FakeVectorStore:
            @classmethod
            def default(cls):
                return cls()

            def all_chunks(self, project_path):
                return [CodeChunk.file_chunk("agent.py", "agent.py#1", "content", 1, 1)]

        with tempfile.TemporaryDirectory() as tmp:
            inputs = iter(["/search Agent run", "exit"])
            output = StringIO()

            with patch("paicli.cli.main.load_api_key", return_value="test-key"), \
                patch("paicli.cli.main.Agent", FakeAgent), \
                patch("paicli.cli.main.PlanExecuteAgent", FakePlanExecuteAgent), \
                patch("paicli.cli.main.AgentOrchestrator", FakeAgentOrchestrator), \
                patch("paicli.cli.main.VectorStore", FakeVectorStore), \
                patch("paicli.cli.main.search_codebase", return_value="🔎 检索: Agent run\n找到 1 个相关代码块"), \
                patch("builtins.input", lambda _: next(inputs)), \
                patch("sys.stdout", output):
                self.assertEqual(main([]), 0)

            self.assertIn("🔎 检索: Agent run", output.getvalue())


if __name__ == "__main__":
    unittest.main()
