"""Interactive command line entrypoint for PaiCli."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, TextIO

from paicli.agent.agent import Agent, AgentEvent
from paicli.agent.agent_orchestrator import AgentOrchestrator
from paicli.agent.plan_execute_agent import PlanExecuteAgent
from paicli.llm.glm_client import GLMClient
from paicli.memory import MemoryManager
from paicli.plan.execution_plan import ExecutionPlan
from paicli.rag import CodeIndex, CodeRetriever, EmbeddingClient, SearchResultFormatter, VectorStore
from paicli.hitl import TerminalHitlHandler
from paicli.tool.hitl_tool_registry import HitlToolRegistry
from paicli.tool.tool_registry import ToolRegistry


BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   ██████╗  █████╗ ██╗      ██████╗██╗     ██╗          ║
║   ██╔══██╗██╔══██╗██║     ██╔════╝██║     ██║          ║
║   ██████╔╝███████║██║     ██║     ██║     ██║          ║
║   ██╔═══╝ ██╔══██║██║     ██║     ██║     ██║          ║
║   ██║     ██║  ██║███████╗╚██████╗███████╗██║          ║
║   ╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝          ║
║           Memory-Enhanced Agent CLI v3.0.0              ║
╚══════════════════════════════════════════════════════════╝
""".strip("\n")
LOGGER = logging.getLogger(__name__)


def format_agent_event(event: AgentEvent) -> Optional[str]:
    if event.type == "thinking":
        return "🤔 思考中..."
    if event.type == "usage":
        return f"📊 Token使用: 输入={event.prompt_tokens}, 输出={event.completion_tokens}"
    if event.type == "tool":
        arguments = json.dumps(event.arguments, ensure_ascii=False, indent=2)
        indented_arguments = _indent_lines(arguments, "   ")
        return (
            f"🔧 执行工具: {event.tool_name}\n"
            f"   参数: {indented_arguments.lstrip()}\n"
            f"   结果: {event.result}"
        )
    return None


def print_agent_event(event: AgentEvent) -> None:
    formatted = format_agent_event(event)
    if formatted:
        print(formatted)
        print()


def _indent_lines(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def format_memory_report(memory_manager: MemoryManager) -> str:
    return "记忆系统状态:\n" + memory_manager.get_system_status()


def format_graph_report(
    name: str,
    vector_store: VectorStore,
    project_path: str | None = None,
) -> str:
    cleaned = name.strip()
    if not cleaned:
        return "⚠️ 用法: /graph 类名或方法名"

    resolved_project_path = project_path or str(Path.cwd().resolve())
    relations = vector_store.find_relations(project_path=resolved_project_path)
    relations = [
        relation for relation in relations
        if relation.source == cleaned
        or relation.target == cleaned
        or relation.source.startswith(f"{cleaned}.")
    ]
    if not relations:
        return f"🕸️ 查询类关系图谱: {cleaned}\n└── 未找到关系"

    lines = [
        f"🕸️ 查询类关系图谱: {cleaned}",
        f"📋 找到 {len(relations)} 条关系:",
        "",
    ]
    grouped: dict[str, list] = {}
    for relation in relations:
        grouped.setdefault(relation.source, []).append(relation)
    for source_index, (source, source_relations) in enumerate(grouped.items()):
        for relation_index, relation in enumerate(source_relations):
            connector = "└──" if relation_index == len(source_relations) - 1 else "├──"
            lines.append(f"   {source} {connector} {relation.relation_type} --> {relation.target}")
    return "\n".join(lines)


def index_codebase(
    path_text: str,
    vector_store: VectorStore,
    tool_registry: ToolRegistry,
    output: TextIO | None = None,
) -> str:
    target = Path(path_text.strip() or ".").expanduser().resolve()
    progress_lines: list[str] = []
    warning_lines: list[str] = []

    def emit(line: str) -> None:
        if output is not None:
            print(line, file=output)

    def on_progress(progress) -> None:
        line = f"进度: {progress.processed_files}/{progress.total_files} ({progress.file_path})"
        progress_lines.append(line)
        emit(line)

    def on_warning(message: str) -> None:
        LOGGER.warning(message)
        line = f"⚠️ {message}"
        warning_lines.append(line)
        emit(line)

    emit("📦 正在索引代码库:")
    emit(f"🔍 开始索引: {target}")

    stats = CodeIndex(
        vector_store=vector_store,
        embedding_client=EmbeddingClient(),
        on_progress=on_progress,
        on_warning=on_warning,
    ).index_with_stats(target)
    tool_registry.set_project_path(str(target))
    lines = [
        "📦 正在索引代码库:",
        f"🔍 开始索引: {target}",
        f"📂 发现 {stats.file_count} 个文件待索引",
        *progress_lines,
        *warning_lines,
        f"✅ 索引完成: {stats.chunk_count} 个代码块, {stats.relation_count} 条关系",
    ]
    if stats.failed_files:
        lines.append(f"⚠️ 跳过 {stats.failed_files} 个解析失败文件")
    report = "\n".join(lines)
    if output is not None:
        for line in lines:
            if line not in {"📦 正在索引代码库:", f"🔍 开始索引: {target}", *progress_lines, *warning_lines}:
                print(line, file=output)
        return ""
    return report


def search_codebase(
    query: str,
    vector_store: VectorStore,
    project_path: str | None = None,
    top_k: int = 5,
) -> str:
    cleaned = query.strip()
    if not cleaned:
        return "⚠️ 用法: /search 自然语言查询"
    resolved_project_path = project_path or str(Path.cwd().resolve())
    try:
        if not vector_store.all_chunks(project_path=resolved_project_path):
            return "代码库尚未索引，请先使用 /index 命令。"
        results = CodeRetriever(vector_store).hybrid_search(
            cleaned,
            top_k=top_k,
            project_path=resolved_project_path,
        )
        return SearchResultFormatter.format_for_cli(cleaned, results)
    except Exception as exc:
        LOGGER.exception("Code search failed")
        return f"检索代码失败: {exc}"


def format_startup_status(api_key_loaded: bool, plan_mode: bool) -> str:
    api_status = "✅ API Key 已加载" if api_key_loaded else "❌ API Key 未加载"
    mode_status = "🧭 使用 Plan-and-Execute 模式" if plan_mode else "🔄 使用 ReAct 模式"
    return f"{api_status}\n{mode_status}"


def save_memory_fact(memory_manager: MemoryManager, fact: str) -> str:
    cleaned = fact.strip()
    if not cleaned:
        return "⚠️ 用法: /save 事实内容"
    memory_manager.save_fact(cleaned, metadata={"source": "manual"})
    return f"💾 已保存事实: {cleaned}"


def clear_agent_history(agent: Agent) -> str:
    facts = agent.clear_history()
    return "\n".join(
        [
            "🧠 提取关键事实到长期记忆...",
            f"提取了 {len(facts)} 条事实",
            "🗑️ 对话历史已清空，关键事实已保存到长期记忆",
        ]
    )


def handle_hitl_command(argument: str, hitl_handler) -> str:
    cleaned = argument.strip().lower()
    if not cleaned:
        status = "开启" if hitl_handler.is_enabled() else "关闭"
        return f"HITL 当前状态: {status}"
    if cleaned == "on":
        hitl_handler.set_enabled(True)
        return "HITL 已启用"
    if cleaned == "off":
        hitl_handler.set_enabled(False)
        return "HITL 已关闭"
    return "⚠️ 用法: /hitl [on|off]"


def finalize_memory_session(memory_manager: MemoryManager) -> None:
    memory_manager.extract_and_save_facts()


def print_banner(output: TextIO = sys.stdout) -> None:
    print(BANNER, file=output)


def load_api_key(env_file: str | Path = ".env") -> Optional[str]:
    path = Path(env_file)
    if path.exists():
        api_key = _read_api_key_from_env_file(path)
        if api_key:
            return api_key
    return os.getenv("GLM_API_KEY") or None


class PlanAgentLike(Protocol):
    def create_plan(self, goal: str) -> ExecutionPlan:
        """Create a plan without executing it."""

    def execute_plan(self, plan: ExecutionPlan) -> str:
        """Execute a previously created plan."""


class TeamAgentLike(Protocol):
    def run(self, user_input: str) -> str:
        """Run a task through the Multi-Agent team."""


def print_plan_update(snapshot: str) -> None:
    print(snapshot)
    print()


def handle_plan_interaction(
    user_input: str,
    agent: PlanAgentLike,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> ExecutionPlan:
    goal = user_input

    while True:
        plan = agent.create_plan(goal)
        print(plan.visualize(), file=output)
        print(file=output)
        answer = input_func("是否执行该计划？(y=执行 / n=取消 / 其他内容=补充要求后重新规划): ").strip()

        if answer.lower() in {"y", "yes", ""}:
            result = agent.execute_plan(plan)
            print(result, file=output)
            print(file=output)
            return plan

        if answer.lower() in {"n", "no", "cancel", "取消"}:
            plan.mark_cancelled("用户取消执行")
            print("已取消执行计划", file=output)
            print(file=output)
            return plan

        goal = f"{user_input}\n补充要求：{answer}"


def handle_team_interaction(
    user_input: str,
    agent: TeamAgentLike,
    output: TextIO = sys.stdout,
) -> str:
    result = agent.run(user_input)
    print(f"🤝 Team: {result}", file=output)
    print(file=output)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    _ = list(argv or [])
    print_banner()

    api_key = load_api_key()
    if not api_key:
        print("❌ 错误: 未找到 GLM_API_KEY", file=sys.stderr)
        return 1

    memory_manager = MemoryManager()
    vector_store: VectorStore | None = None
    llm_client = GLMClient(api_key)
    hitl_handler = TerminalHitlHandler(False)
    tool_registry = HitlToolRegistry(hitl_handler)

    def handle_agent_event(event: AgentEvent) -> None:
        print_agent_event(event)

    agent = Agent(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        on_event=handle_agent_event,
        memory_manager=memory_manager,
    )
    plan_agent = PlanExecuteAgent(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        on_plan_update=print_plan_update,
        memory_manager=memory_manager,
    )
    team_agent = AgentOrchestrator(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        on_event=lambda message: print(message),
    )
    plan_mode = False
    team_mode = False
    print(format_startup_status(api_key_loaded=True, plan_mode=plan_mode))
    print()
    print("💡 提示:")
    print("   - 输入你的问题或任务")
    print("   - 输入 '/plan' 进入计划模式")
    print("   - 输入 '/team' 后，下一条任务使用 Multi-Agent 团队模式")
    print("   - 输入 '/hitl [on|off]' 查看或切换人工审批")
    print("   - 输入 '/memory' 查看记忆和 Token 状态")
    print("   - 输入 '/index [路径]' 索引代码库")
    print("   - 输入 '/search 查询' 检索代码库")
    print("   - 输入 '/graph 名称' 查询代码关系图谱")
    print("   - 输入 '/save 事实内容' 手动保存长期记忆")
    print("   - 输入 '/clear' 清空对话历史")
    print("   - 输入 'exit' 或 'quit' 退出\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            finalize_memory_session(memory_manager)
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            finalize_memory_session(memory_manager)
            break
        if user_input.lower() == "/plan":
            plan_mode = True
            print("🧭 已进入计划模式，请输入复杂任务。\n")
            continue
        if user_input.lower() == "/team":
            team_mode = True
            print("🤝 已进入 Multi-Agent 团队模式，下一条任务将由规划者、执行者、检查者协作完成。\n")
            continue
        if user_input.lower().startswith("/hitl"):
            print(handle_hitl_command(user_input[len("/hitl") :], hitl_handler))
            print()
            continue
        if user_input.lower() == "/memory":
            print(format_memory_report(memory_manager))
            print()
            continue
        if user_input.lower().startswith("/index"):
            if vector_store is None:
                vector_store = VectorStore.default()
            report = index_codebase(user_input[len("/index") :], vector_store, tool_registry, output=sys.stdout)
            if report:
                print(report)
            print()
            continue
        if user_input.lower().startswith("/search"):
            if vector_store is None:
                vector_store = VectorStore.default()
            print(search_codebase(user_input[len("/search") :], vector_store, tool_registry.project_path))
            print()
            continue
        if user_input.lower().startswith("/graph"):
            if vector_store is None:
                vector_store = VectorStore.default()
            print(format_graph_report(user_input[len("/graph") :], vector_store, tool_registry.project_path))
            print()
            continue
        if user_input.lower().startswith("/save"):
            print(save_memory_fact(memory_manager, user_input[len("/save") :]))
            print()
            continue
        if user_input.lower() in {"clear", "/clear"}:
            print(clear_agent_history(agent))
            hitl_handler.clear_approved_all()
            print()
            continue

        if plan_mode:
            handle_plan_interaction(user_input, plan_agent)
            plan_mode = False
        elif team_mode:
            handle_team_interaction(user_input, team_agent, output=sys.stdout)
            team_mode = False
        else:
            response = agent.run(user_input)
            print(f"🤖 Agent: {response}\n")

    return 0


def _read_api_key_from_env_file(path: Path) -> Optional[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key.strip() == "GLM_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
