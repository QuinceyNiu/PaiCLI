import tempfile
import time
import unittest
import json
from pathlib import Path

from paicli.agent.plan_execute_agent import PlanExecuteAgent
from paicli.plan.planner import PLANNING_PROMPT
from paicli.memory import LongTermMemory, MemoryManager
from paicli.plan.execution_plan import ExecutionPlan, PlanStatus
from paicli.plan.task import Task, TaskStatus, TaskType
from paicli.tool.tool_registry import ToolExecutionResult, ToolRegistry


class FakePlanner:
    def __init__(self, plan):
        self.plan = plan
        self.goals = []

    def create_plan(self, goal):
        self.goals.append(goal)
        return self.plan


class FakeReactAgent:
    def __init__(self):
        self.inputs = []

    def run(self, user_input):
        self.inputs.append(user_input)
        return "react result"


class FakeMemoryManager:
    def __init__(self):
        self.saved_facts = []
        self.context_queries = []
        self.tool_results = []
        self.extracted = False

    def build_context_for_query(self, query, limit=5):
        self.context_queries.append((query, limit))
        return "[相关记忆]\n- FACT: 用户喜欢 JDK 17"

    def add_tool_result(self, tool_name, result):
        self.tool_results.append((tool_name, result))

    def extract_and_save_facts(self):
        self.extracted = True
        return ["执行计划完成"]


class SlowBatchToolRegistry:
    def __init__(self):
        self.batches = []

    def execute_tools(self, invocations):
        self.batches.append(list(invocations))
        import concurrent.futures

        def run(invocation):
            time.sleep(float(json.loads(invocation.arguments_json)["delay"]))
            return ToolExecutionResult.completed(
                invocation,
                f"done:{invocation.id}",
                1,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(invocations)) as executor:
            return list(executor.map(run, invocations))


def temp_memory_manager(tmp: str) -> MemoryManager:
    return MemoryManager(long_term_memory=LongTermMemory(storage_dir=Path(tmp) / "memory"))


class PlanExecuteAgentTest(unittest.TestCase):
    def test_should_plan_uses_action_keywords_and_length(self) -> None:
        agent = PlanExecuteAgent(
            api_key="test-key",
            planner=FakePlanner(ExecutionPlan("plan", "goal")),
            react_agent=FakeReactAgent(),
        )

        self.assertFalse(agent.should_plan("读 README"))
        self.assertTrue(agent.should_plan("创建项目，然后写接口，接着执行测试"))
        self.assertTrue(agent.should_plan("x" * 51))

    def test_planning_prompt_guides_code_understanding_tasks_to_search_code(self) -> None:
        self.assertIn("search_code", PLANNING_PROMPT)
        self.assertIn("代码理解", PLANNING_PROMPT)

    def test_simple_task_delegates_to_react_agent(self) -> None:
        react_agent = FakeReactAgent()
        planner = FakePlanner(ExecutionPlan("plan", "goal"))
        agent = PlanExecuteAgent(api_key="test-key", planner=planner, react_agent=react_agent)

        result = agent.run("读 README")

        self.assertEqual(result, "react result")
        self.assertEqual(react_agent.inputs, ["读 README"])
        self.assertEqual(planner.goals, [])

    def test_create_plan_returns_plan_without_executing(self) -> None:
        plan = ExecutionPlan("plan_preview", "预览计划")
        task = Task(
            "task_1",
            "写入文件",
            TaskType.FILE_WRITE,
            arguments={"path": "hello.txt", "content": "hello"},
        )
        plan.add_task(task)
        self.assertTrue(plan.compute_execution_order())
        agent = PlanExecuteAgent(
            api_key="test-key",
            planner=FakePlanner(plan),
            react_agent=FakeReactAgent(),
        )

        preview = agent.create_plan("创建文件，然后写内容，接着预览")

        self.assertIs(preview, plan)
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_complex_task_creates_plan_and_executes_tasks_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = ExecutionPlan("plan_1", "创建文件然后读取")
            write = Task(
                "task_1",
                "写入 hello.txt",
                TaskType.FILE_WRITE,
                arguments={"path": "hello.txt", "content": "hello pai"},
            )
            read = Task(
                "task_2",
                "读取 hello.txt",
                TaskType.FILE_READ,
                dependencies=["task_1"],
                arguments={"path": "hello.txt"},
            )
            plan.add_task(write)
            plan.add_task(read)
            self.assertTrue(plan.compute_execution_order())

            agent = PlanExecuteAgent(
                api_key="test-key",
                planner=FakePlanner(plan),
                react_agent=FakeReactAgent(),
                tool_registry=ToolRegistry(base_dir=tmp),
                memory_manager=temp_memory_manager(tmp),
            )

            result = agent.run("创建文件，然后写内容，接着读取")

            self.assertEqual(plan.status, PlanStatus.COMPLETED)
            self.assertEqual(write.status, TaskStatus.COMPLETED)
            self.assertEqual(read.status, TaskStatus.COMPLETED)
            self.assertEqual((Path(tmp) / "hello.txt").read_text(encoding="utf-8"), "hello pai")
            self.assertIn("执行计划", result)
            self.assertIn("全部任务执行完成", result)
            self.assertIn("task_1", result)
            self.assertIn("文件内容:\nhello pai", read.result)

    def test_execute_plan_runs_an_existing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = ExecutionPlan("plan_execute", "执行已有计划")
            task = Task(
                "task_1",
                "写入 hello.txt",
                TaskType.FILE_WRITE,
                arguments={"path": "hello.txt", "content": "hello"},
            )
            plan.add_task(task)
            self.assertTrue(plan.compute_execution_order())
            agent = PlanExecuteAgent(
                api_key="test-key",
                planner=FakePlanner(plan),
                react_agent=FakeReactAgent(),
                tool_registry=ToolRegistry(base_dir=tmp),
                memory_manager=temp_memory_manager(tmp),
            )

            result = agent.execute_plan(plan)

            self.assertEqual(plan.status, PlanStatus.COMPLETED)
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertIn("全部任务执行完成", result)

    def test_execute_plan_uses_memory_context_and_extracts_facts_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = ExecutionPlan("plan_memory", "执行记忆计划")
            task = Task(
                "task_1",
                "写入 hello.txt",
                TaskType.FILE_WRITE,
                arguments={"path": "hello.txt", "content": "hello"},
            )
            plan.add_task(task)
            self.assertTrue(plan.compute_execution_order())
            memory_manager = FakeMemoryManager()
            agent = PlanExecuteAgent(
                api_key="test-key",
                planner=FakePlanner(plan),
                react_agent=FakeReactAgent(),
                tool_registry=ToolRegistry(base_dir=tmp),
                memory_manager=memory_manager,
            )

            agent.execute_plan(plan)

            self.assertEqual(memory_manager.context_queries, [("写入 hello.txt", 5)])
            self.assertEqual(memory_manager.tool_results[0][0], "write_file")
            self.assertTrue(memory_manager.extracted)

    def test_execute_plan_runs_same_dependency_batch_in_parallel(self) -> None:
        plan = ExecutionPlan("plan_parallel", "并行读取")
        first = Task(
            "task_1",
            "读取 A",
            TaskType.FILE_READ,
            arguments={"path": "a.txt", "delay": "0.2"},
        )
        second = Task(
            "task_2",
            "读取 B",
            TaskType.FILE_READ,
            arguments={"path": "b.txt", "delay": "0.2"},
        )
        final = Task("task_3", "汇总", TaskType.ANALYSIS, dependencies=["task_1", "task_2"])
        plan.add_task(first)
        plan.add_task(second)
        plan.add_task(final)
        self.assertTrue(plan.compute_execution_order())
        registry = SlowBatchToolRegistry()
        agent = PlanExecuteAgent(
            api_key="test-key",
            planner=FakePlanner(plan),
            react_agent=FakeReactAgent(),
            tool_registry=registry,
            memory_manager=FakeMemoryManager(),
        )

        started = time.perf_counter()
        agent.execute_plan(plan)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.35)
        self.assertEqual([[inv.id for inv in batch] for batch in registry.batches], [["task_1", "task_2"]])
        self.assertEqual(first.status, TaskStatus.COMPLETED)
        self.assertEqual(second.status, TaskStatus.COMPLETED)
        self.assertEqual(final.status, TaskStatus.COMPLETED)


    def test_complex_task_emits_plan_snapshots_as_status_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = ExecutionPlan("plan_live", "实时展示计划")
            write = Task(
                "task_1",
                "写入 hello.txt",
                TaskType.FILE_WRITE,
                arguments={"path": "hello.txt", "content": "hello pai"},
            )
            read = Task(
                "task_2",
                "读取 hello.txt",
                TaskType.FILE_READ,
                dependencies=["task_1"],
                arguments={"path": "hello.txt"},
            )
            plan.add_task(write)
            plan.add_task(read)
            self.assertTrue(plan.compute_execution_order())
            snapshots = []
            agent = PlanExecuteAgent(
                api_key="test-key",
                planner=FakePlanner(plan),
                react_agent=FakeReactAgent(),
                tool_registry=ToolRegistry(base_dir=tmp),
                on_plan_update=snapshots.append,
                memory_manager=temp_memory_manager(tmp),
            )

            agent.run("创建文件，然后写内容，接着读取")

            self.assertGreaterEqual(len(snapshots), 5)
            self.assertIn("⏳ task_1", snapshots[0])
            self.assertTrue(any("▶️ task_1" in snapshot for snapshot in snapshots))
            self.assertTrue(any("✅ task_1" in snapshot for snapshot in snapshots))
            self.assertIn("✅ task_2", snapshots[-1])

    def test_failed_task_marks_plan_failed_and_skips_dependents(self) -> None:
        plan = ExecutionPlan("plan_2", "失败计划")
        read_missing = Task(
            "task_1",
            "读取不存在文件",
            TaskType.FILE_READ,
            arguments={"path": "missing.txt"},
        )
        dependent = Task(
            "task_2",
            "分析读取结果",
            TaskType.ANALYSIS,
            dependencies=["task_1"],
        )
        plan.add_task(read_missing)
        plan.add_task(dependent)
        self.assertTrue(plan.compute_execution_order())

        with tempfile.TemporaryDirectory() as tmp:
            agent = PlanExecuteAgent(
                api_key="test-key",
                planner=FakePlanner(plan),
                react_agent=FakeReactAgent(),
                tool_registry=ToolRegistry(base_dir=tmp),
                memory_manager=temp_memory_manager(tmp),
            )

            result = agent.run("创建项目，然后读取文件，接着分析结果")

        self.assertEqual(plan.status, PlanStatus.FAILED)
        self.assertEqual(read_missing.status, TaskStatus.FAILED)
        self.assertEqual(dependent.status, TaskStatus.SKIPPED)
        self.assertIn("任务执行失败", result)
        self.assertIn("读取文件失败", result)


if __name__ == "__main__":
    unittest.main()
