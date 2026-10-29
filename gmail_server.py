"""MCP server that exposes the user's Gmail inbox as tools.

Runs as its own process. An MCP client (the Mosaic agent, or any other MCP client) starts
it as a subprocess and talks to it over stdin/stdout using JSON-RPC (the "stdio" transport).
The client discovers the tools with list_tools, so the schema below is all it needs to know.
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

import tools

server = MCPServer(
    name="mosaic-gmail",
    instructions="Read-only access to the student's Gmail inbox.",
)


# The MCP SDK builds the tool's JSON Schema from the type hints: `str` becomes "string", a
# default value makes the argument optional, and `Field(description=...)` becomes the
# argument's description. `description` below is the text the LLM reads to pick this tool.
@server.tool(
    description=(
        "Search the user's Gmail inbox by keywords. Mails matching more keywords rank "
        "higher; ties are newest first. Results include labels: mails labelled "
        "CATEGORY_PROMOTIONS are usually ads, but can still hold important notices. "
        "Use 1-3 specific keywords per topic (e.g. 'razorpay interview'). To look up "
        "several independent topics (e.g. DBMS, OS lab, ML project), call this tool "
        "several times in the SAME turn, one topic per call, instead of one per turn. "
        "Deadlines are written as dates in mails, so to find what is due on a day, "
        "search for the date itself (e.g. '26 October')."
    )
)
def search_emails(
    query: Annotated[str, Field(description="Space-separated keywords, 1-3 per topic.")],
    max_results: Annotated[int, Field(description="Max mails to return.")] = 5,
) -> list[dict]:
    """Search via tools.search_emails and tag each mail with the evidence fields.

    Every MCP server returns items with `source` and `title` (plus the existing `id`, `url`
    and `date`). The agent's citation registry relies only on these shared fields, so it
    never needs to know what a mail or a Notion row looks like inside.
    """
    return [{"source": "gmail", "title": mail["subject"], **mail} for mail in tools.search_emails(query, max_results)]


if __name__ == "__main__":
    server.run()  # stdio transport by default
