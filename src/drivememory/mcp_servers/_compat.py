"""Import the MCP server class for both MCP Python SDK v2 (MCPServer) and v1 (FastMCP)."""

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server  # type: ignore

__all__ = ["Server"]
