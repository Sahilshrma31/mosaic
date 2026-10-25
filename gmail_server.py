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
        "several times in the SAME turn, one topic per call, instead of one per turn."
    )
)
def search_emails(
    query: Annotated[str, Field(description="Space-separated keywords, 1-3 per topic.")],
    max_results: Annotated[int, Field(description="Max mails to return.")] = 5,
) -> list[dict]:
    """Delegate to the mock search; swapping in the Gmail API only changes tools.search_emails."""
    return tools.search_emails(query, max_results)


if __name__ == "__main__":
    server.run()  # stdio transport by default
