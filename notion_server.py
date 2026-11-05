"""MCP server that exposes the user's Notion databases as tools.

Runs as its own process and talks to its client over stdin/stdout (the "stdio" transport),
exactly like gmail_server.py.
"""

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field

import tools

server = MCPServer(
    name="mosaic-notion",
    instructions="The student's Notion tracker and study planner. Writes need the student's approval.",
)


# The type hints become the JSON Schema: `Literal[...]` becomes an enum, so the LLM can only
# send one of the two database names, and the `None` default makes `status` optional.
@server.tool(
    description=(
        "Read the user's Notion databases. 'applications' is the internship tracker "
        "(company, role, status: Applied/Interview/Rejected/Offer, applied date, notes). "
        "'tasks' is the study planner (task, subject, due date, status: Not Started/"
        "In Progress/Done); it also holds interview-prep tasks, so check it for anything "
        "about an interview. Notion can be out of date, so cross-check with emails."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
def query_notion(
    database: Literal["applications", "tasks"],
    status: Annotated[str | None, Field(description="Optional status filter, case-insensitive.")] = None,
) -> list[dict]:
    """Query via tools.query_notion and tag each row with the evidence fields (see gmail_server.py).

    The title is the row's name column (company or task) and the date is its key date
    (applied date or due date), so citations read like "Notion: Swiggy, 2026-09-22".
    """
    return [
        {
            "source": "notion",
            "title": row.get("company") or row.get("task", ""),
            "date": row.get("applied_date") or row.get("due_date"),
            **row,
        }
        for row in tools.query_notion(database, status)
    ]


# Not marked read-only, so the agent never offers this tool to the LLM (see ToolHub). It only
# runs after the student clicks "Yes" on a proposed update.
@server.tool(
    description="Change the status of one application in the tracker.",
    annotations=ToolAnnotations(read_only_hint=False),
)
def update_application_status(
    row_id: str,
    status: Literal["Applied", "Interview", "Rejected", "Offer"],
) -> list[dict]:
    """Returns [{"saved": bool}]; saved is False on sample data, which never changes."""
    return [{"saved": tools.update_application_status(row_id, status)}]


if __name__ == "__main__":
    server.run()  # stdio transport by default
