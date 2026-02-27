"""AIR Blackbox MCP Server — EU AI Act compliance scanning for AI agents."""

__version__ = "0.1.2"


def main():
    """Entry point for the MCP server."""
    from .server import mcp
    mcp.run()
