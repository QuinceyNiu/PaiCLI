"""Static approval policy for mutating PaiCli tools."""

from __future__ import annotations


class ApprovalPolicy:
    DANGEROUS_TOOLS = frozenset({"write_file", "execute_command", "create_project"})

    @classmethod
    def requires_approval(cls, tool_name: str) -> bool:
        return tool_name in cls.DANGEROUS_TOOLS or cls.is_mcp_tool(tool_name)

    @staticmethod
    def is_mcp_tool(tool_name: str) -> bool:
        return bool(tool_name and tool_name.startswith("mcp__"))

    @staticmethod
    def get_danger_level(tool_name: str) -> str:
        if tool_name == "execute_command":
            return "🔴 高危"
        if tool_name in {"write_file", "create_project"}:
            return "🟡 中危"
        if ApprovalPolicy.is_mcp_tool(tool_name):
            return "🟡 中危"
        return "🟢 安全"

    @staticmethod
    def get_risk_description(tool_name: str) -> str:
        if tool_name == "execute_command":
            return "将在系统上执行 Shell 命令，可能修改文件、安装软件或影响系统状态"
        if tool_name == "write_file":
            return "将写入或覆盖文件内容，原有内容将丢失"
        if tool_name == "create_project":
            return "将在磁盘上创建新目录和文件"
        if ApprovalPolicy.is_mcp_tool(tool_name):
            return "将调用第三方 MCP 工具，工具行为取决于外部 server"
        return "安全的只读操作"
