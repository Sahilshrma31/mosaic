"""MCP server that exposes the user's Notion databases as tools.

Runs as its own process and talks to its client over stdin/stdout (the "stdio" transport),
exactly like gmail_server.py.
"""

from mcp.server.mcpserver import MCPServer

import tools

server = MCPServer(
    name="mosaic-notion",
    instructions="Read-only access to the student's Notion tracker and study planner.",
)


# `status: str | None = None` makes status optional, so the generated schema only
# requires `database`. The description is what the LLM reads to pick arguments.
@server.tool(
    description=(
        "Read the user's Notion databases. 'applications' is the internship tracker "
        "(company, role, status: Applied/Interview/Rejected/Offer, applied date, notes). "
        "'tasks' is the study planner (task, subject, due date, status: Not Started/"
        "In Progress/Done). Notion can be out of date, so cross-check with emails."
    )
)
def query_notion(database: str, status: str | None = None) -> list[dict]:
    """Delegate to the mock Notion query; swapping in the Notion API only changes tools.query_notion."""
    return tools.query_notion(database, status)


if __name__ == "__main__":
    server.run()  # stdio transport by default
