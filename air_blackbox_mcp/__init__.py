"""EU AI Act Compliance MCP Server — scanning and remediation for AI agents."""

__version__ = "1.0.0"


def main():
    """Entry point for the MCP server."""
    from .server import mcp
    mcp.run()
