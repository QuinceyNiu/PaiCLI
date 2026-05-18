import unittest

from paicli.plan.task import Task, TaskStatus, TaskType
from paicli.plan.execution_plan import ExecutionPlan, PlanStatus


class TaskModelTest(unittest.TestCase):
    def test_task_starts_pending_and_records_dependencies(self) -> None:
        task = Task(
            id="write_api",
            description="写 REST API",
            type=TaskType.FILE_WRITE,
            dependencies=["create_project"],
        )

        self.assertEqual(task.id, "write_api")
        self.assertEqual(task.description, "写 REST API")
        self.assertEqual(task.type, TaskType.FILE_WRITE)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.dependencies, ["create_project"])
        self.assertEqual(task.dependents, [])
        self.assertIsNone(task.result)
        self.assertIsNone(task.error)

    def test_task_lifecycle_records_status_result_error_and_timestamps(self) -> None:
        task = Task(id="package", description="打包项目", type=TaskType.COMMAND)

        task.mark_started()

        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertGreater(task.start_time, 0)

        task.mark_completed("build success")

        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, "build success")
        self.assertGreaterEqual(task.end_time, task.start_time)

        failed = Task(id="verify", description="验证结果", type=TaskType.VERIFICATION)
        failed.mark_started()
        failed.mark_failed("test failed")

        self.assertEqual(failed.status, TaskStatus.FAILED)
        self.assertEqual(failed.error, "test failed")
        self.assertGreaterEqual(failed.end_time, failed.start_time)

    def test_task_is_executable_only_when_pending_and_dependencies_completed(self) -> None:
        create_project = Task(
            id="create_project",
            description="创建项目",
            type=TaskType.FILE_WRITE,
        )
        write_api = Task(
            id="write_api",
            description="写 API",
            type=TaskType.FILE_WRITE,
            dependencies=["create_project"],
        )
        tasks = {task.id: task for task in [create_project, write_api]}

        self.assertFalse(write_api.is_executable(tasks))

        create_project.mark_started()
        create_project.mark_completed("created")

        self.assertTrue(write_api.is_executable(tasks))

        write_api.mark_started()

        self.assertFalse(write_api.is_executable(tasks))

    def test_missing_dependency_makes_task_not_executable(self) -> None:
        task = Task(
            id="run",
            description="运行应用",
            type=TaskType.COMMAND,
            dependencies=["package"],
        )

        self.assertFalse(task.is_executable({task.id: task}))


class ExecutionPlanModelTest(unittest.TestCase):
    def test_compute_execution_order_sorts_tasks_by_dependencies(self) -> None:
        plan = ExecutionPlan(id="plan_1", goal="创建并运行 Spring Boot API")
        create_project = Task("create_project", "创建项目", TaskType.FILE_WRITE)
        write_api = Task("write_api", "写 API", TaskType.FILE_WRITE, ["create_project"])
        package_app = Task("package", "打包项目", TaskType.COMMAND, ["write_api"])
        run_app = Task("run", "运行应用", TaskType.COMMAND, ["package"])

        plan.add_task(run_app)
        plan.add_task(package_app)
        plan.add_task(write_api)
        plan.add_task(create_project)

        self.assertTrue(plan.compute_execution_order())
        self.assertEqual(
            plan.execution_order,
            ["create_project", "write_api", "package", "run"],
        )
        self.assertEqual(create_project.dependents, ["write_api"])
        self.assertEqual(write_api.dependents, ["package"])

    def test_compute_execution_order_returns_false_for_cycles(self) -> None:
        plan = ExecutionPlan(id="plan_cycle", goal="错误依赖")
        plan.add_task(Task("a", "A", TaskType.ANALYSIS, ["c"]))
        plan.add_task(Task("b", "B", TaskType.ANALYSIS, ["a"]))
        plan.add_task(Task("c", "C", TaskType.ANALYSIS, ["b"]))

        self.assertFalse(plan.compute_execution_order())
        self.assertEqual(plan.execution_order, [])

    def test_plan_lifecycle_reflects_task_failures(self) -> None:
        plan = ExecutionPlan(id="plan_2", goal="验证应用")
        task = Task("verify", "验证结果", TaskType.VERIFICATION)
        plan.add_task(task)

        self.assertEqual(plan.status, PlanStatus.CREATED)

        plan.mark_started()
        self.assertEqual(plan.status, PlanStatus.RUNNING)
        self.assertGreater(plan.start_time, 0)

        task.mark_started()
        task.mark_failed("assertion failed")

        self.assertTrue(plan.has_failed())

        plan.mark_failed("verification failed")
        self.assertEqual(plan.status, PlanStatus.FAILED)
        self.assertEqual(plan.summary, "verification failed")
        self.assertGreaterEqual(plan.end_time, plan.start_time)

    def test_plan_can_be_completed_or_cancelled(self) -> None:
        plan = ExecutionPlan(id="plan_3", goal="简单任务")

        plan.mark_started()
        plan.mark_completed("all done")

        self.assertEqual(plan.status, PlanStatus.COMPLETED)
        self.assertEqual(plan.summary, "all done")

        cancelled = ExecutionPlan(id="plan_4", goal="取消任务")
        cancelled.mark_started()
        cancelled.mark_cancelled("user cancelled")

        self.assertEqual(cancelled.status, PlanStatus.CANCELLED)
        self.assertEqual(cancelled.summary, "user cancelled")

    def test_visualize_uses_status_icons(self) -> None:
        plan = ExecutionPlan(id="plan_5", goal="展示计划")
        pending = Task("task_1", "等待任务", TaskType.FILE_READ)
        running = Task("task_2", "运行任务", TaskType.COMMAND)
        completed = Task("task_3", "完成任务", TaskType.ANALYSIS)
        failed = Task("task_4", "失败任务", TaskType.VERIFICATION)
        skipped = Task("task_5", "跳过任务", TaskType.FILE_WRITE)

        running.mark_started()
        completed.mark_started()
        completed.mark_completed("ok")
        failed.mark_started()
        failed.mark_failed("boom")
        skipped.mark_skipped("blocked")

        for task in [pending, running, completed, failed, skipped]:
            plan.add_task(task)
        self.assertTrue(plan.compute_execution_order())

        output = plan.visualize()

        self.assertIn("⏳ task_1", output)
        self.assertIn("▶️ task_2", output)
        self.assertIn("✅ task_3", output)
        self.assertIn("❌ task_4", output)
        self.assertIn("⏭️ task_5", output)


if __name__ == "__main__":
    unittest.main()
