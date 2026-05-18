import json
import unittest

from paicli.llm.glm_client import ChatResponse, Message
from paicli.plan.planner import Planner, PlanningError
from paicli.plan.task import Task, TaskStatus, TaskType
from paicli.plan.execution_plan import ExecutionPlan


class FakePlannerLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.responses.pop(0)


class PlannerTest(unittest.TestCase):
    def test_create_plan_calls_llm_once_and_parses_plan(self) -> None:
        llm = FakePlannerLLM(
            [
                ChatResponse(
                    message=Message.assistant(
                        json.dumps(
                            {
                                "summary": "创建 REST API 并运行",
                                "tasks": [
                                    {
                                        "id": "create",
                                        "description": "创建项目结构",
                                        "type": "FILE_WRITE",
                                        "dependencies": [],
                                    },
                                    {
                                        "id": "api",
                                        "description": "编写 REST API",
                                        "type": "FILE_WRITE",
                                        "dependencies": ["create"],
                                    },
                                    {
                                        "id": "run",
                                        "description": "运行应用",
                                        "type": "COMMAND",
                                        "dependencies": ["api"],
                                    },
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        planner = Planner(llm)

        plan = planner.create_plan("创建一个 REST API 并运行")

        self.assertEqual(len(llm.calls), 1)
        messages, tools = llm.calls[0]
        self.assertIsNone(tools)
        self.assertEqual(messages[0].role, "system")
        self.assertIn("请按以下JSON格式输出执行计划", messages[0].content)
        self.assertIn("创建一个 REST API 并运行", messages[1].content)
        self.assertEqual(plan.goal, "创建一个 REST API 并运行")
        self.assertEqual(plan.summary, "创建 REST API 并运行")
        self.assertEqual(plan.execution_order, ["task_1", "task_2", "task_3"])
        self.assertEqual(plan.tasks["task_2"].dependencies, ["task_1"])
        self.assertEqual(plan.tasks["task_3"].type, TaskType.COMMAND)

    def test_create_plan_parses_task_arguments(self) -> None:
        llm = FakePlannerLLM(
            [
                ChatResponse(
                    message=Message.assistant(
                        json.dumps(
                            {
                                "summary": "写文件",
                                "tasks": [
                                    {
                                        "id": "write",
                                        "description": "写入文件",
                                        "type": "FILE_WRITE",
                                        "dependencies": [],
                                        "arguments": {
                                            "path": "hello.txt",
                                            "content": "hello pai",
                                        },
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        planner = Planner(llm)

        plan = planner.create_plan("写 hello.txt")

        self.assertEqual(
            plan.tasks["task_1"].arguments,
            {"path": "hello.txt", "content": "hello pai"},
        )

    def test_parse_plan_strips_markdown_and_remaps_duplicate_ids(self) -> None:
        llm = FakePlannerLLM(
            [
                ChatResponse(
                    message=Message.assistant(
                        """```json
{
  "summary": "处理重复 ID",
  "tasks": [
    {"id": "task", "description": "第一步", "type": "FILE_READ", "dependencies": []},
    {"id": "task", "description": "第二步", "type": "ANALYSIS", "dependencies": ["task"]}
  ]
}
```"""
                    )
                )
            ]
        )
        planner = Planner(llm)

        plan = planner.create_plan("处理重复 ID")

        self.assertEqual(list(plan.tasks), ["task_1", "task_2"])
        self.assertEqual(plan.tasks["task_2"].dependencies, ["task_1"])
        self.assertEqual(plan.execution_order, ["task_1", "task_2"])

    def test_create_plan_rejects_invalid_json_and_cycles(self) -> None:
        invalid_json_planner = Planner(FakePlannerLLM([ChatResponse(message=Message.assistant("不是 JSON"))]))

        with self.assertRaises(PlanningError):
            invalid_json_planner.create_plan("坏计划")

        cycle_planner = Planner(
            FakePlannerLLM(
                [
                    ChatResponse(
                        message=Message.assistant(
                            json.dumps(
                                {
                                    "summary": "循环依赖",
                                    "tasks": [
                                        {
                                            "id": "a",
                                            "description": "A",
                                            "type": "ANALYSIS",
                                            "dependencies": ["b"],
                                        },
                                        {
                                            "id": "b",
                                            "description": "B",
                                            "type": "ANALYSIS",
                                            "dependencies": ["a"],
                                        },
                                    ],
                                }
                            )
                        )
                    )
                ]
            )
        )

        with self.assertRaisesRegex(PlanningError, "循环依赖"):
            cycle_planner.create_plan("循环计划")

    def test_replan_includes_completed_tasks_and_failure_reason(self) -> None:
        failed_plan = ExecutionPlan(id="plan_old", goal="创建 API")
        completed = Task("task_1", "创建项目", TaskType.FILE_WRITE)
        completed.mark_started()
        completed.mark_completed("项目已创建")
        failed = Task("task_2", "运行测试", TaskType.VERIFICATION, ["task_1"])
        failed.mark_started()
        failed.mark_failed("测试失败")
        failed_plan.add_task(completed)
        failed_plan.add_task(failed)

        llm = FakePlannerLLM(
            [
                ChatResponse(
                    message=Message.assistant(
                        json.dumps(
                            {
                                "summary": "修复后重新验证",
                                "tasks": [
                                    {
                                        "id": "fix",
                                        "description": "修复测试失败",
                                        "type": "FILE_WRITE",
                                        "dependencies": [],
                                    },
                                    {
                                        "id": "verify",
                                        "description": "重新运行测试",
                                        "type": "VERIFICATION",
                                        "dependencies": ["fix"],
                                    },
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        planner = Planner(llm)

        plan = planner.replan(failed_plan, "测试失败")

        self.assertEqual(plan.summary, "修复后重新验证")
        user_prompt = llm.calls[0][0][1].content
        self.assertIn("原始目标：创建 API", user_prompt)
        self.assertIn("已完成任务", user_prompt)
        self.assertIn("创建项目: 项目已创建", user_prompt)
        self.assertIn("失败原因：测试失败", user_prompt)


if __name__ == "__main__":
    unittest.main()
