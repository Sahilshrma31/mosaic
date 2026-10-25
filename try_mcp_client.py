"""Connect to each MCP server as a client: list its tools, then call one. No LLM involved."""

import asyncio
import sys

from mcp import Client, StdioServerParameters

# (server script, tool to call, arguments, field to print from each result row)
CHECKS = [
    ("gmail_server.py", "search_emails", {"query": "swiggy application"}, "subject"),
    ("notion_server.py", "query_notion", {"database": "applications", "status": "applied"}, "company"),
]


async def check(script: str, tool_name: str, args: dict, field: str) -> None:
    """Start one server as a subprocess, print the tools it advertises, and call one of them."""
    # Same Python as this script (the .venv one). Leaving `async with` shuts the server down.
    server = StdioServerParameters(command=sys.executable, args=[script])

    async with Client(server) as client:
        print(f"=== {script}")
        listed = await client.list_tools()
        for tool in listed.tools:
            print("Tool discovered:", tool.name)
            print("  required args:", tool.input_schema.get("required"))

        result = await client.call_tool(tool_name, args)
        print(f"Result of {tool_name}({args}):")
        for row in result.structured_content["result"]:
            print(" ", row["id"], "|", row[field])
        print()


async def main() -> None:
    for script, tool_name, args, field in CHECKS:
        await check(script, tool_name, args, field)


if __name__ == "__main__":
    asyncio.run(main())
