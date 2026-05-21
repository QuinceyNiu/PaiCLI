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
from paicli.hitl import TerminalHitlHandler
from paicli.llm.glm_client import GLMClient
from paicli.memory import MemoryManager
from paicli.mcp import (
    DEFAULT_MCP_CONFIG_PATH,
    PROJECT_MCP_CONFIG_PATH,
    BrowserSession,
    McpConfig,
    McpServerManager,
    McpToolProvider,
    ensure_default_mcp_config,
)
from paicli.plan.execution_plan import ExecutionPlan
from paicli.rag import CodeIndex, CodeRetriever, EmbeddingClient, SearchResultFormatter, VectorStore
from paicli.skill import SkillRegistry
from paicli.tool.hitl_tool_registry import HitlToolRegistry
from paicli.tool.hitl_tool_registry import BrowserGuard
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


def format_mcp_status(server_names: list[str]) -> str:
    if not server_names:
        return "🔌 MCP server：未配置"
    return "🔌 MCP server：" + ", ".join(server_names)


def format_mcp_table(manager: McpServerManager) -> str:
    runtimes = manager.runtimes()
    if not runtimes:
        return "🔌 MCP server：未配置"
    lines = ["🔌 MCP server 状态:"]
    for runtime in runtimes:
        status = {
            "ready": "● ready",
            "error": "✗ error",
            "disabled": "○ disabled",
        }.get(runtime.status, runtime.status)
        uptime = _format_uptime(runtime.uptime_seconds)
        pid = f" | PID {runtime.pid}" if runtime.pid else ""
        error = f" | {runtime.error}" if runtime.error else ""
        lines.append(
            f"{runtime.config.name}: {status} | {runtime.config.transport} | "
            f"{len(runtime.tools)} tools | {uptime}{pid}{error}"
        )
    return "\n".join(lines)


def handle_mcp_command(argument: str, manager: McpServerManager, registry: ToolRegistry) -> str:
    parts = argument.strip().split()
    if not parts:
        return format_mcp_table(manager)
    command = parts[0].lower()
    name = parts[1] if len(parts) > 1 else ""
    if command in {"restart", "enable", "disable", "logs"} and not name:
        return f"⚠️ 用法: /mcp {command} <name>"
    if command == "restart":
        result = manager.restart(name)
        refresh_mcp_tools(registry, manager)
        return result
    if command == "enable":
        result = manager.enable(name)
        refresh_mcp_tools(registry, manager)
        return result
    if command == "disable":
        result = manager.disable(name)
        refresh_mcp_tools(registry, manager)
        return result
    if command == "logs":
        return manager.logs(name)
    return "⚠️ 用法: /mcp [restart|logs|disable|enable] <name>"


def handle_browser_command(argument: str, session: BrowserSession) -> str:
    command = argument.strip().lower()
    if command in {"", "status"}:
        return session.status()
    if command in {"connect", "shared"}:
        return session.switch_to_shared()
    if command in {"disconnect", "isolated"}:
        return session.disconnect()
    return "⚠️ 用法: /browser [status|connect|disconnect]"


def handle_skill_command(argument: str, registry: SkillRegistry) -> str:
    parts = argument.strip().split()
    if not parts:
        return "⚠️ 用法: /skill [list|show|on|off|reload] <name>"
    command = parts[0].lower()
    name = parts[1] if len(parts) > 1 else ""
    if command == "list":
        return registry.format_list()
    if command == "show":
        if not name:
            return "⚠️ 用法: /skill show <name>"
        skill = registry.get(name)
        if skill is None:
            return f"未找到 Skill: {name}"
        return f"# {skill.name}\n来源: {skill.source.value}\n路径: {skill.path}\n\n{skill.document}"
    if command == "off":
        if not name:
            return "⚠️ 用法: /skill off <name>"
        if not registry.disable(name):
            return f"未找到 Skill: {name}"
        return f"已禁用 Skill: {name}"
    if command == "on":
        if not name:
            return "⚠️ 用法: /skill on <name>"
        if not registry.enable(name):
            return f"未找到 Skill: {name}"
        return f"已启用 Skill: {name}"
    if command == "reload":
        registry.reload()
        return "Skills 重新加载完成。\n" + registry.format_summary()
    return "⚠️ 用法: /skill [list|show|on|off|reload] <name>"


def refresh_mcp_tools(registry: ToolRegistry, manager: McpServerManager) -> None:
    if not manager.runtimes():
        return
    registry.unregister_by_prefix("mcp__")
    for tool in McpToolProvider(manager=manager).get_tools():
        registry.register(tool)


def _format_uptime(seconds: int) -> str:
    if seconds <= 0:
        return "0s"
    minutes, remainder = divmod(seconds, 60)
    if minutes <= 0:
        return f"{remainder}s"
    hours, minutes = divmod(minutes, 60)
    if hours <= 0:
        return f"{minutes}m"
    return f"{hours}h{minutes}m"


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


def format_plan_summary(plan: ExecutionPlan) -> str:
    batches = plan.get_execution_batches()
    first_batch = batches[0] if batches else []
    executable = [task for task in plan.tasks.values() if task.is_executable(plan.tasks)]
    final_batch = batches[-1] if batches else []

    lines = [
        "📋 计划摘要",
        f"- 目标: {plan.goal}",
        f"- 任务数: {len(plan.tasks)} | 并行批次: {len(batches)} | 当前可执行: {len(executable)} | 状态: {plan.status.value}",
    ]
    if first_batch:
        lines.append("- 首批执行: " + ", ".join(task.id for task in first_batch))
    if final_batch and final_batch != first_batch:
        lines.append("- 最终收敛: " + ", ".join(task.id for task in final_batch))
    return "\n".join(lines)


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
        print(format_plan_summary(plan), file=output)
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
    skill_registry = SkillRegistry(project_dir=Path.cwd() / ".paicli" / "skills")
    skill_registry.reload()
    print(skill_registry.format_summary())
    hitl_handler = TerminalHitlHandler(False)
    mcp_bootstrap_message = ensure_default_mcp_config(DEFAULT_MCP_CONFIG_PATH)
    if mcp_bootstrap_message:
        print(mcp_bootstrap_message)
    mcp_config = McpConfig.load_merged(
        user_path=DEFAULT_MCP_CONFIG_PATH,
        project_path=PROJECT_MCP_CONFIG_PATH,
        project_dir=Path.cwd(),
    )
    mcp_manager = McpServerManager(mcp_config)
    mcp_manager.start_all(progress_callback=print)
    browser_session = BrowserSession(mcp_manager, hitl_handler=hitl_handler)
    browser_guard = BrowserGuard(browser_session)
    tool_registry = HitlToolRegistry(hitl_handler, browser_guard=browser_guard)
    tool_registry.on_browser_restart = lambda: refresh_mcp_tools(tool_registry, mcp_manager)
    refresh_mcp_tools(tool_registry, mcp_manager)

    def handle_agent_event(event: AgentEvent) -> None:
        print_agent_event(event)

    agent = Agent(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        on_event=handle_agent_event,
        memory_manager=memory_manager,
        skill_registry=skill_registry,
    )
    plan_agent = PlanExecuteAgent(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        on_plan_update=print_plan_update,
        memory_manager=memory_manager,
        skill_registry=skill_registry,
    )
    team_agent = AgentOrchestrator(
        api_key,
        llm_client=llm_client,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        on_event=lambda message: print(message),
        skill_registry=skill_registry,
    )
    plan_mode = False
    team_mode = False
    print(format_startup_status(api_key_loaded=True, plan_mode=plan_mode))
    print(format_mcp_status(mcp_config.server_names()))
    print()
    print("💡 提示:")
    print("   - 输入你的问题或任务")
    print("   - 输入 '/plan' 进入计划模式，或 '/plan 任务' 直接规划任务")
    print("   - 输入 '/team' 后，下一条任务使用 Multi-Agent 团队模式")
    print("   - 输入 '/hitl [on|off]' 查看或切换人工审批")
    print("   - 输入 '/mcp' 查看 MCP server 状态")
    print("   - 输入 '/browser status' 查看浏览器模式，'/browser disconnect' 切回 isolated")
    print("   - 输入 '/skill list' 查看 Skill，'/skill reload' 热加载 Skill")
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
        if user_input.lower().startswith("/plan "):
            plan_goal = user_input[len("/plan") :].strip()
            if not plan_goal:
                plan_mode = True
                print("🧭 已进入计划模式，请输入复杂任务。\n")
                continue
            print("🧭 使用 Plan-and-Execute 模式")
            print()
            handle_plan_interaction(plan_goal, plan_agent, input_func=input, output=sys.stdout)
            plan_mode = False
            continue
        if user_input.lower() == "/team":
            team_mode = True
            print("🤝 已进入 Multi-Agent 团队模式，下一条任务将由规划者、执行者、检查者协作完成。\n")
            continue
        if user_input.lower().startswith("/hitl"):
            print(handle_hitl_command(user_input[len("/hitl") :], hitl_handler))
            print()
            continue
        if user_input.lower().startswith("/mcp"):
            print(handle_mcp_command(user_input[len("/mcp") :], mcp_manager, tool_registry))
            print()
            continue
        if user_input.lower().startswith("/browser"):
            print(handle_browser_command(user_input[len("/browser") :], browser_session))
            refresh_mcp_tools(tool_registry, mcp_manager)
            print()
            continue
        if user_input.lower().startswith("/skill"):
            print(handle_skill_command(user_input[len("/skill") :], skill_registry))
            agent.refresh_skill_index()
            if hasattr(plan_agent.react_agent, "refresh_skill_index"):
                plan_agent.react_agent.refresh_skill_index()
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
            react_agent = getattr(plan_agent, "react_agent", None)
            if hasattr(react_agent, "clear_history"):
                react_agent.clear_history()
            hitl_handler.clear_approved_all()
            print()
            continue

        if plan_mode:
            handle_plan_interaction(user_input, plan_agent, input_func=input, output=sys.stdout)
            plan_mode = False
        elif team_mode:
            handle_team_interaction(user_input, team_agent, output=sys.stdout)
            team_mode = False
        else:
            response = agent.run(user_input)
            print(f"🤖 Agent: {response}\n")

    mcp_manager.close()
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
