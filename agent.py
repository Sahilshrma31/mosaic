"""Hand-written agent loop: the LLM asks for tool calls, we run them, and repeat until it answers.

The agent is an MCP client. It never imports the tools; it discovers them from the MCP
servers listed in SERVERS and sends every tool call to the server that owns that tool.
"""

import asyncio
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from mcp import Client, StdioServerParameters

import memory

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

# Tried in order. Each model has its own free-tier quota (20 requests/day), so this chain gives
# ~80 requests a day instead of 20. The lite models were 10x slower under daytime load, so
# they're last resorts. All four were checked to emit correct tool calls with our MCP tools.
MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# Public demo mode (MOSAIC_DEMO=1), for the deployed site that strangers can use. Visitors must
# not write to long-term memory, so the `remember` tool isn't offered and any call to it is refused.
DEMO_MODE = os.getenv("MOSAIC_DEMO") == "1"

# Upper bound on LLM round trips per question. Without it, a model that keeps asking for
# tools would loop forever (and burn through the free-tier rate limit).
MAX_STEPS = 6

# {today} and {facts} are filled in per request by _system_prompt(), so a server that runs for
# days still knows the right date, and newly remembered facts apply to the very next question.
SYSTEM_PROMPT = """You are Mosaic, an assistant for a college student.
Answer questions using the student's Gmail and Notion data, which you can read with the tools.
Today's date is {today}.
Known facts about the student (long-term memory):
{facts}
Rules:
- Always look the facts up with the tools before answering. Never invent dates, names or statuses.
- For a question that needs several facts (e.g. "everything about my Razorpay interview"),
  first call `plan` with the sub-questions, in the same turn as your first searches.
  Split by what the student needs to know (when and where, format, prep planned or still
  pending, current status), not by data source. Then cover every sub-question with the
  right tools, which often means both Notion databases plus Gmail.
  Skip `plan` for simple questions.
- Batch independent lookups: request all the tool calls you need in one turn.
- If two sources disagree, trust the most recent email and point out the mismatch.
- Ignore promotional mails unless they contain an action the student must take.
- Every tool result item has a `cite` number. Put it in square brackets right after each
  claim it supports, e.g. "Round 1 is on 28 Oct at 3 PM [3]." Cite only numbers you were
  given, and never write URLs.
- Cite the item that actually states the fact. When an email shows a Notion status is out
  of date, cite the email for the real status and the Notion row only for what Notion says,
  e.g. "Rejected [4]; your tracker still says Applied [2]." Never state a status or date
  that none of your evidence contains.
- Facts from long-term memory have no cite number; say "you told me" instead.
- Compare every date with today's date: say clearly when a deadline is today, tomorrow or
  already past, and put anything due today first.
- For questions about a day or period (today, tomorrow, this week), also search emails for
  the dates themselves, e.g. "26 October": mails state deadlines as dates and rarely use the
  word "deadline", so a topic search alone misses them.
- When the student states a lasting fact or preference about themselves (goals, preferences,
  decisions, e.g. "Google is my dream company"), call `remember` with it as one short
  third-person sentence. Don't save one-off or temporary things, or facts already listed above.
- Earlier turns of this conversation are included for context; use them to resolve
  follow-ups like "what about Round 2?", but look facts up again rather than trusting them.
- Keep the final answer short and clear. Use **bold** and "- " bullet lists when they help;
  no tables."""


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

    def __init__(self, data_mode: str | None = None) -> None:
        """`data_mode` ("mock" or "live") overrides MOSAIC_DATA from .env for the servers.

        The evals pass "mock" so they always run on the data with known answers, whatever
        .env says. It works because the SDK merges `env` into the subprocess environment
        and load_dotenv never overrides a variable that is already set.
        """
        self._stack = AsyncExitStack()
        self._route: dict[str, Client] = {}  # tool name -> the server connection that owns it
        self._env = {"MOSAIC_DATA": data_mode} if data_mode else None
        self.declarations: list[types.FunctionDeclaration] = []

    async def start(self) -> None:
        for script in SERVERS:
            server = StdioServerParameters(command=sys.executable, args=[str(BASE_DIR / script)], env=self._env)
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


# A tool the agent handles itself instead of sending to an MCP server. It runs nothing: it only
# records the model's sub-questions, so decomposition costs no extra LLM call and the model
# can call it in the same turn as its first searches.
PLAN_DECLARATION = types.FunctionDeclaration(
    name="plan",
    description=(
        "Record the sub-questions you will answer before searching. Use it only for questions "
        "that need several facts, in the same turn as your first searches."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
        "required": ["sub_questions"],
    },
)

# Another local tool: saves a long-term fact to SQLite. Like `plan`, it costs no extra LLM call;
# the model calls it alongside its other tool calls when the student states a lasting preference.
REMEMBER_DECLARATION = types.FunctionDeclaration(
    name="remember",
    description="Save a lasting fact or preference about the student to long-term memory.",
    parameters_json_schema={
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "One short third-person sentence."}},
        "required": ["fact"],
    },
)

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [3] or [1, 2]


class EvidenceRegistry:
    """Numbers every item the tools return, so the LLM cites [n] while the code owns n -> source.

    The model never writes a URL or a source title itself. It only sees each item with a
    `cite` number and repeats that number in its answer; the mapping from number to the real
    mail or Notion page stays here. A citation the model invents (a number that was never
    handed out) is dropped, so it can't fabricate a source.
    """

    def __init__(self) -> None:
        self.items: list[dict] = []
        self._numbers: dict[tuple, int] = {}  # (source, id) -> citation number

    def tag(self, result: dict) -> dict:
        """Give every item in a tool result its citation number (adds a `cite` field).

        The same mail returned by two searches keeps the same number, so the model never
        sees one source under two different numbers.
        """
        if "result" not in result:
            return result  # an error message: nothing to cite
        tagged = []
        for item in result["result"]:
            key = (item.get("source"), item.get("id") or item.get("url"))
            if key not in self._numbers:
                self.items.append(item)
                self._numbers[key] = len(self.items)
            tagged.append({"cite": self._numbers[key], **item})
        return {"result": tagged}

    def finalize(self, answer: str) -> tuple[str, list[dict]]:
        """Validate the answer's citations and renumber them 1, 2, 3... in reading order.

        The model may cite [7] and [12] out of 15 items it saw; the user should see [1] and [2]
        with exactly two sources listed. "[1, 2]" is rewritten as "[1][2]", and numbers that
        match no evidence are removed. Returns the cleaned answer and the cited sources.
        """
        renumber: dict[int, int] = {}  # number the model used -> number the user sees

        def replace(match: re.Match) -> str:
            kept = []
            for raw in match.group(1).split(","):
                old = int(raw)
                if 1 <= old <= len(self.items):
                    renumber.setdefault(old, len(renumber) + 1)
                    kept.append(renumber[old])
            return "".join(f"[{n}]" for n in dict.fromkeys(kept))

        text = re.sub(r" +([.,;])", r"\1", CITATION.sub(replace, answer))  # no " ." left by a dropped cite
        sources = []
        for old, new in renumber.items():
            item = self.items[old - 1]
            sources.append(
                {"n": new, "id": item.get("id"), "source": item.get("source", ""),
                 "title": item.get("title", ""), "url": item.get("url"), "date": item.get("date")}
            )
        return text, sources


@dataclass
class AgentReply:
    """Everything one question produced: the answer, the sources it cites, and how it got there."""

    answer: str
    sources: list[dict] = field(default_factory=list)  # [{n, id, source, title, url, date}]
    plan: list[str] = field(default_factory=list)  # sub-questions, if the model decomposed
    trace: list[dict] = field(default_factory=list)  # [{round, call, result}] per tool call
    model: str = ""  # which model in MODELS produced the answer ("" if none did)


def _system_prompt(today: date | None = None) -> str:
    """The system prompt with today's date and the long-term facts filled in (read fresh each time).

    `today` is normally the real date; the evals pin it so date questions ("what's due
    today?") have the same correct answer on every run.
    """
    facts = memory.all_facts()
    return SYSTEM_PROMPT.format(
        today=(today or date.today()).isoformat(),
        facts="\n".join(f"- {fact}" for fact in facts) if facts else "- (none yet)",
    )


def _configs(hub: ToolHub, today: date | None = None) -> tuple[types.GenerateContentConfig, types.GenerateContentConfig]:
    """Build the normal config (tools on) and the last-step config (tools off) for one question.

    Built per question, not once at import, because the system prompt carries today's date
    and the current long-term facts. The last allowed step switches tool calls off (mode
    NONE), so the model must answer from the evidence it already gathered.
    """
    system_prompt = _system_prompt(today)
    local_tools = [PLAN_DECLARATION] if DEMO_MODE else [PLAN_DECLARATION, REMEMBER_DECLARATION]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(function_declarations=[*local_tools, *hub.declarations])],
    )
    final_step_config = config.model_copy(
        update={
            "system_instruction": system_prompt
            + "\n- You cannot call tools any more. Answer now with the evidence you already have,"
            " and briefly say what you could not check.",
            "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        }
    )
    return config, final_step_config


def _history_contents(conversation_id: str | None) -> list[types.Content]:
    """Earlier turns of this conversation as Gemini messages (short-term memory).

    Only the questions and final answers are replayed, not the tool calls and results behind
    them: those are large, and the model can search again if it needs the details. Citation
    markers are stripped, because their numbers belonged to that turn's evidence list and
    would clash with the new one.
    """
    if not conversation_id:
        return []
    return [
        types.Content(
            role="user" if role == "user" else "model",
            parts=[types.Part.from_text(text=re.sub(r" +([.,;])", r"\1", CITATION.sub("", text)))],
        )
        for role, text in memory.recent_history(conversation_id)
    ]


async def run_agent(
    question: str, hub: ToolHub, conversation_id: str | None = None, today: date | None = None
) -> AgentReply:
    """Answer one question with memory, moving down MODELS when a model fails transiently.

    Memory: the conversation's recent turns are loaded before the loop, and the question
    and answer are saved after it, so the next question in the same conversation can build
    on them. Nothing is saved when every model failed, so error messages never become history.

    Two errors trigger the next model:
    - 5xx: Google's side is busy (503) or timed out (504), so another model is likely to work.
    - 429: the free-tier quota is used up. Quotas are counted per model (20 requests/day each),
      so the next model may still have quota left.
    The question is restarted from scratch on the next model rather than continued, so one
    conversation never mixes two models. Other 4xx errors (e.g. a malformed request) would
    fail the same way on any model, so they are raised instead.
    """
    history = _history_contents(conversation_id)
    for model in MODELS:
        try:
            reply = await _run_loop(question, model, hub, history, today)
        except errors.ServerError as exc:
            print(f"[{model}] server error {exc.code}, trying the next model...")
            continue
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            print(f"[{model}] free-tier quota used up (429), trying the next model...")
            continue
        if conversation_id:
            memory.add_message(conversation_id, "user", question)
            memory.add_message(conversation_id, "assistant", reply.answer)
        return reply
    return AgentReply("Mosaic's AI models are overloaded or out of free quota right now. Please try again later.")


async def _run_loop(
    question: str, model: str, hub: ToolHub, history: list[types.Content], today: date | None = None
) -> AgentReply:
    """Answer one question by looping between the LLM and the MCP tools.

    Each round sends the whole conversation so far (earlier turns, the question, every
    tool call and every tool result) because the LLM is stateless and remembers nothing
    between requests. If the reply contains function calls, we run them, append the results
    and loop again. A reply with no function calls is the final answer.

    Every tool result passes through the EvidenceRegistry, which numbers each item before
    the model sees it; the final answer's [n] citations are then checked against it. The
    local tools are answered here, not by an MCP server: `plan` keeps the sub-questions and
    `remember` saves a long-term fact.

    The loop is capped at MAX_STEPS rounds so a confused model can't loop forever. The last
    round runs with tools disabled, which forces a best-effort answer from the evidence
    gathered so far (graceful degradation) instead of an empty failure.

    The Gemini call uses the async client (client.aio) so that, inside FastAPI, waiting for
    the model doesn't block other requests.
    """
    config, final_step_config = _configs(hub, today)
    contents = [*history, types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    evidence = EvidenceRegistry()
    plan: list[str] = []
    trace: list[dict] = []

    for step in range(1, MAX_STEPS + 1):
        step_config = final_step_config if step == MAX_STEPS else config
        response = await client.aio.models.generate_content(model=model, contents=contents, config=step_config)

        if not response.function_calls:
            print(f"[step {step}] final answer from {model}, no tool calls")
            answer, sources = evidence.finalize(response.text or "")
            return AgentReply(answer, sources, plan, trace, model)

        # Keep the model's own turn (its function calls) in the history, unchanged.
        contents.append(response.candidates[0].content)

        # Run every requested tool and send all the results back in one turn.
        result_parts = []
        for call in response.function_calls:
            args = call.args or {}
            if call.name == "plan":
                plan = [str(q) for q in args.get("sub_questions", [])]
                result = {"result": "Plan noted. Answer every sub-question, citing the evidence."}
            elif call.name == "remember" and DEMO_MODE:
                result = {"error": "Long-term memory is turned off in the public demo."}
            elif call.name == "remember":
                saved = memory.remember(str(args.get("fact", "")))
                result = {"result": "Saved to long-term memory." if saved else "Already known."}
            else:
                result = evidence.tag(await hub.call(call.name, args))
            summary = _summary(result)
            print(f"[step {step}] {call.name}({args}) -> {summary}")
            trace.append({"round": step, "call": f"{call.name}({_format_args(args)})", "result": summary})
            result_parts.append(
                types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result))
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return AgentReply("Sorry, I couldn't finish answering within the step limit. Try a more specific question.",
                      plan=plan, trace=trace, model=model)


def _format_args(args: dict) -> str:
    """Render tool arguments compactly for the trace, e.g. database="tasks", status="done"."""
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _summary(result: dict) -> str:
    """One-line description of a tool result for the trace."""
    if "error" in result:
        return f"ERROR {result['error']}"
    if isinstance(result["result"], list):
        return f"{len(result['result'])} results"
    return "noted"


async def _main() -> None:
    """Ask one question from the terminal: start the MCP servers, answer, shut them down."""
    question = input("Ask Mosaic: ")
    hub = ToolHub()
    await hub.start()
    try:
        reply = await run_agent(question, hub)
        if reply.plan:
            print("\nPlan:", *reply.plan, sep="\n  - ")
        print("\n" + reply.answer)
        for source in reply.sources:
            print(f"  [{source['n']}] {source['source']}: {source['title']} ({source['url']})")
    finally:
        await hub.stop()


if __name__ == "__main__":
    asyncio.run(_main())
