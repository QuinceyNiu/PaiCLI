"""MCP client support for PaiCli."""

from paicli.mcp.client import McpClient, McpHttpClient, McpStdioClient, McpTool
from paicli.mcp.browser_session import BrowserMode, BrowserSession
from paicli.mcp.config import (
    DEFAULT_CHROME_DEVTOOLS_SERVER,
    DEFAULT_MCP_CONFIG_PATH,
    DEFAULT_MCP_TEMPLATE,
    PROJECT_MCP_CONFIG_PATH,
    McpConfig,
    McpServerConfig,
    ensure_default_mcp_config,
)
from paicli.mcp.manager import McpServerManager, McpServerRuntime
from paicli.mcp.schema import McpSchemaSanitizer
from paicli.mcp.tool_provider import McpToolProvider, normalize_tool_name
from paicli.mcp.transport import McpTransport, StdioTransport, StreamableHttpTransport

__all__ = [
    "DEFAULT_CHROME_DEVTOOLS_SERVER",
    "DEFAULT_MCP_CONFIG_PATH",
    "DEFAULT_MCP_TEMPLATE",
    "PROJECT_MCP_CONFIG_PATH",
    "BrowserMode",
    "BrowserSession",
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
    "ensure_default_mcp_config",
    "normalize_tool_name",
]
