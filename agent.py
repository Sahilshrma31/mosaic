"""Hand-written agent loop: the LLM asks for tool calls, we run them, and repeat until it answers.

The agent is an MCP client. It never imports the tools; it discovers them from the MCP
servers listed in SERVERS and sends every tool call to the server that owns that tool.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from mcp import Client, StdioServerParameters

load_dotenv()

BASE_DIR = Path(__file__).parent

# One MCP server per data source. Adding a source (e.g. Google Calendar) means writing a new
# server and adding its script here; the loop below doesn't change.
SERVERS = ["gmail_server.py", "notion_server.py"]

# The SDK silently retries failed requests with growing waits, which can take minutes on an
# overloaded model. Fail fast instead (60s timeout, 2 attempts) so the fallback model can step in.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=2)),
)

MODEL = "gemini-3.5-flash"
# Used when MODEL is overloaded (503) or out of free quota (429).
FALLBACK_MODEL = "gemini-3.6-flash"

# Upper bound on LLM round trips per question. Without it, a model that keeps asking for
# tools would loop forever (and burn through the free-tier rate limit).
MAX_STEPS = 6

SYSTEM_PROMPT = f"""You are Mosaic, an assistant for a college student.
Answer questions using the student's Gmail and Notion data, which you can read with the tools.
Today's date is {date.today().isoformat()}.
Rules:
- Always look the facts up with the tools before answering. Never invent dates, names or statuses.
- Batch independent lookups: request all the tool calls you need in one turn.
- If two sources disagree, trust the most recent email and point out the mismatch.
- Ignore promotional mails unless they contain an action the student must take.
- Keep the final answer short and clear."""


class ToolHub:
    """Keeps one open MCP connection per server and routes each tool call to the right one.

    start() launches every script in SERVERS as a subprocess, asks each server for its tools
    (list_tools), and converts every MCP tool into a Gemini function declaration. MCP and
    Gemini describe a tool with the same three things (name, description, JSON Schema for
    the arguments), so the conversion is a direct copy.

    The connections stay open until stop(). Normally `async with Client(...)` closes the
    connection at the end of its block; the AsyncExitStack keeps all of them open together
    and closes them all at once, so a question never pays the cost of starting processes.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._route: dict[str, Client] = {}  # tool name -> the server connection that owns it
        self.declarations: list[types.FunctionDeclaration] = []

    async def start(self) -> None:
        for script in SERVERS:
            server = StdioServerParameters(command=sys.executable, args=[str(BASE_DIR / script)])
            connection = await self._stack.enter_async_context(Client(server))
            for tool in (await connection.list_tools()).tools:
                if tool.name in self._route:
                    raise RuntimeError(f"Tool '{tool.name}' is offered by more than one MCP server")
                self._route[tool.name] = connection
                self.declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.input_schema,
                    )
                )
        print(f"[mcp] connected to {len(SERVERS)} servers, tools: {list(self._route)}")

    async def call(self, name: str, args: dict) -> dict:
        """Run one tool on its MCP server and wrap the outcome in a dict for the LLM to read.

        Never raises. An unknown tool, an error raised inside the tool (the server reports it
        with is_error) or a broken connection all come back as {"error": "..."}, so the LLM
        sees what went wrong and can retry instead of the whole request crashing.
        """
        connection = self._route.get(name)
        if connection is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {list(self._route)}"}
        try:
            result = await connection.call_tool(name, args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if result.is_error:
            return {"error": " ".join(getattr(block, "text", "") for block in result.content)}
        return {"result": (result.structured_content or {}).get("result", [])}

    async def stop(self) -> None:
        await self._stack.aclose()


def _configs(hub: ToolHub) -> tuple[types.GenerateContentConfig, types.GenerateContentConfig]:
    """Build the normal config (tools on) and the last-step config (tools off) from the hub's tools.

    The last allowed step switches tool calls off (mode NONE), so the model must answer from
    the evidence it already gathered instead of the user getting no answer at all.
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=hub.declarations)],
    )
    final_step_config = config.model_copy(
        update={
            "system_instruction": SYSTEM_PROMPT
            + "\n- You cannot call tools any more. Answer now with the evidence you already have,"
            " and briefly say what you could not check.",
            "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        }
    )
    return config, final_step_config


async def run_agent(question: str, hub: ToolHub) -> str:
    """Answer one question, switching to FALLBACK_MODEL if MODEL fails for a transient reason.

    Two errors trigger the fallback:
    - 5xx: Google's side is busy (503) or timed out (504), so another model is likely to work.
    - 429: the free-tier quota is used up. Quotas are counted per model (20 requests/day each),
      so the fallback model may still have quota left.
    The question is restarted from scratch on the fallback model rather than continued, so
    one conversation never mixes two models. Other 4xx errors (e.g. a malformed request)
    would fail the same way on any model, so they are raised instead.
    """
    for model in (MODEL, FALLBACK_MODEL):
        try:
            return await _run_loop(question, model, hub)
        except errors.ServerError as exc:
            print(f"[{model}] server error {exc.code}, trying the next model...")
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            print(f"[{model}] free-tier quota used up (429), trying the next model...")
    return "Mosaic's AI models are overloaded or out of free quota right now. Please try again later."


async def _run_loop(question: str, model: str, hub: ToolHub) -> str:
    """Answer one question by looping between the LLM and the MCP tools.

    Each round sends the whole conversation so far (the question, every earlier tool call
    and every tool result) because the LLM is stateless and remembers nothing between
    requests. If the reply contains function calls, we run them, append the results and
    loop again. A reply with no function calls is the final answer.

    The loop is capped at MAX_STEPS rounds so a confused model can't loop forever. The last
    round runs with tools disabled, which forces a best-effort answer from the evidence
    gathered so far (graceful degradation) instead of an empty failure.

    The Gemini call uses the async client (client.aio) so that, inside FastAPI, waiting for
    the model doesn't block other requests.
    """
    config, final_step_config = _configs(hub)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for step in range(1, MAX_STEPS + 1):
        step_config = final_step_config if step == MAX_STEPS else config
        response = await client.aio.models.generate_content(model=model, contents=contents, config=step_config)

        if not response.function_calls:
            print(f"[step {step}] final answer, no tool calls")
            return response.text

        # Keep the model's own turn (its function calls) in the history, unchanged.
        contents.append(response.candidates[0].content)

        # Run every requested tool and send all the results back in one turn.
        result_parts = []
        for call in response.function_calls:
            result = await hub.call(call.name, call.args or {})
            print(f"[step {step}] {call.name}({call.args}) -> {_summary(result)}")
            result_parts.append(
                types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result))
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "Sorry, I couldn't finish answering within the step limit. Try a more specific question."


def _summary(result: dict) -> str:
    """One-line description of a tool result for the console trace."""
    if "error" in result:
        return f"ERROR {result['error']}"
    return f"{len(result['result'])} results"


async def _main() -> None:
    """Ask one question from the terminal: start the MCP servers, answer, shut them down."""
    question = input("Ask Mosaic: ")
    hub = ToolHub()
    await hub.start()
    try:
        print(await run_agent(question, hub))
    finally:
        await hub.stop()


if __name__ == "__main__":
    asyncio.run(_main())
