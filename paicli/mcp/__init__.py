"""MCP client support for PaiCli."""

from paicli.mcp.client import McpClient, McpHttpClient, McpStdioClient, McpTool
from paicli.mcp.config import DEFAULT_MCP_CONFIG_PATH, PROJECT_MCP_CONFIG_PATH, McpConfig, McpServerConfig
from paicli.mcp.manager import McpServerManager, McpServerRuntime
from paicli.mcp.schema import McpSchemaSanitizer
from paicli.mcp.tool_provider import McpToolProvider, normalize_tool_name
from paicli.mcp.transport import McpTransport, StdioTransport, StreamableHttpTransport

__all__ = [
    "DEFAULT_MCP_CONFIG_PATH",
    "PROJECT_MCP_CONFIG_PATH",
    "McpClient",
    "McpConfig",
    "McpHttpClient",
    "McpSchemaSanitizer",
    "McpServerManager",
    "McpServerRuntime",
    "McpServerConfig",
    "McpStdioClient",
    "McpTool",
    "McpToolProvider",
    "McpTransport",
    "StdioTransport",
    "StreamableHttpTransport",
    "normalize_tool_name",
]
