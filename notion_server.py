"""MCP server that exposes the user's Notion databases as tools.

Runs as its own process and talks to its client over stdin/stdout (the "stdio" transport),
exactly like gmail_server.py.
"""

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

import tools

server = MCPServer(
    name="mosaic-notion",
    instructions="Read-only access to the student's Notion tracker and study planner.",
)


# The type hints become the JSON Schema: `Literal[...]` becomes an enum, so the LLM can only
# send one of the two database names, and the `None` default makes `status` optional.
@server.tool(
    description=(
        "Read the user's Notion databases. 'applications' is the internship tracker "
        "(company, role, status: Applied/Interview/Rejected/Offer, applied date, notes). "
        "'tasks' is the study planner (task, subject, due date, status: Not Started/"
        "In Progress/Done). Notion can be out of date, so cross-check with emails."
    )
)
def query_notion(
    database: Literal["applications", "tasks"],
    status: Annotated[str | None, Field(description="Optional status filter, case-insensitive.")] = None,
) -> list[dict]:
    """Delegate to the mock Notion query; swapping in the Notion API only changes tools.query_notion."""
    return tools.query_notion(database, status)


if __name__ == "__main__":
    server.run()  # stdio transport by default
