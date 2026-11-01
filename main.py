import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import memory
import tools
from agent import ToolHub, run_agent

# Pass the app's MOSAIC_DATA (from the environment, or .env loaded by agent.py) on to the MCP
# servers explicitly: the SDK gives subprocesses only a minimal environment, so a value set on
# the host (e.g. MOSAIC_DATA=mock on Render) would otherwise never reach them.
hub = ToolHub(data_mode=os.getenv("MOSAIC_DATA"))
DATA_MODE = "live" if os.getenv("MOSAIC_DATA") == "live" else "mock"


def _today() -> date | None:
    """The date the agent treats as today: MOSAIC_TODAY if set, else 24 Oct 2026 on mock data.

    On live data it's None, meaning the real date. On mock data a real date would make every
    planted deadline look long past, so the public demo would seem broken a week after launch.
    """
    if os.getenv("MOSAIC_TODAY"):
        return date.fromisoformat(os.environ["MOSAIC_TODAY"])
    return tools.MOCK_TODAY if DATA_MODE == "mock" else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the memory tables and start the MCP servers once when the app boots.

    Code before `yield` runs at startup, code after it at shutdown. Every request in
    between reuses the same open connections instead of launching new server processes.
    """
    memory.init()
    await hub.start()
    yield
    await hub.stop()


app = FastAPI(title="Mosaic", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_PAGE = STATIC_DIR / "index.html"
CHAT_PAGE = STATIC_DIR / "chat.html"

# "no-cache" doesn't mean "never cache": the browser keeps its copy but asks the server on
# every load whether the file changed (a cheap 304 reply if not). Without it, the browser kept
# running an old chat.js after a rebuild, so new features silently didn't show up.
NO_CACHE = {"Cache-Control": "no-cache"}


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that adds Cache-Control: no-cache to every response."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


# Serves the compiled frontend (chat.js and home.js, built from frontend/*.ts with `npm run build`).
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Serve the landing page."""
    return FileResponse(INDEX_PAGE, headers=NO_CACHE)


@app.get("/chat")
def chat_page():
    """Serve the full-screen chat app (GET; asking a question is POST /chat/stream). Served from
    the same server as the API (same origin), so the page's fetch() calls need no CORS setup."""
    return FileResponse(CHAT_PAGE, headers=NO_CACHE)


@app.get("/config")
def config() -> dict:
    """What the chat page needs to describe itself: sample or live data, and the pinned "today".

    e.g. {"data": "mock", "demo": true, "today": "2026-10-24"}; today is null on live data.
    """
    today = _today()
    return {"data": DATA_MODE, "demo": agent.DEMO_MODE, "today": today.isoformat() if today else None}


class ChatRequest(BaseModel):
    """Request body for POST /chat, e.g. {"question": "What about Round 2?", "conversation_id": "..."}.

    FastAPI validates incoming JSON against this model and rejects a body without a
    `question` string with a 422 error, before any LLM call is made. `conversation_id` is
    created by the page for each chat; questions with the same id share short-term memory.
    """

    question: str = Field(max_length=1000)  # long enough for any real question, caps token waste
    conversation_id: str | None = Field(default=None, max_length=64)


class Source(BaseModel):
    """One cited source. `n` matches the [n] in the answer text."""

    n: int
    id: str | None = None  # the mail / Notion page id (mock data: msg_00x, app_00x, task_00x)
    source: str  # "gmail" or "notion"
    title: str
    url: str | None = None
    date: str | None = None


class TraceStep(BaseModel):
    """One tool call the agent made, e.g. round 1, search_emails(query='swiggy'), 2 results."""

    round: int
    call: str
    result: str


class ChatResponse(BaseModel):
    """Response body for POST /chat.

    e.g. {"answer": "Rejected [1]. Notion still says Applied [2].", "sources": [...],
    "plan": [], "trace": [...]}. The page turns each [n] into a link to sources[n].
    """

    answer: str
    sources: list[Source] = []
    plan: list[str] = []
    trace: list[TraceStep] = []
    model: str = ""  # the Gemini model that produced the answer


# The public demo runs on a free Gemini quota (~80 requests a day, 2-4 per question), so it
# answers at most this many questions per day; after that it explains instead of failing.
DEMO_DAILY_LIMIT = int(os.getenv("MOSAIC_DEMO_DAILY_LIMIT", "15"))
DEMO_LIMIT_MESSAGE = (
    "The public demo has answered its {limit} questions for today: it runs on a free AI quota. "
    "Please come back tomorrow."
)
_demo_usage = {"day": "", "count": 0}


def _take_demo_slot() -> bool:
    """Count one question against today's demo limit. Returns False once the limit is reached.

    The counter lives in memory, so it resets when Render restarts the free instance; it's a
    soft guard against someone burning the quota in one sitting, while the separate demo API
    key's own quota is the hard limit. Check-and-increment has no `await` in between, so two
    concurrent requests can't both take the last slot.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    if _demo_usage["day"] != today:
        _demo_usage.update(day=today, count=0)
    if _demo_usage["count"] >= DEMO_DAILY_LIMIT:
        return False
    _demo_usage["count"] += 1
    return True


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer the user's question with the agent loop over their Gmail and Notion data.

    This route only handles HTTP: validating the request and shaping the response.
    All agent logic lives in agent.py, so the agent can change without touching this file.
    It's `async def` because run_agent awaits the LLM and the MCP servers; while one
    question waits on them, the server keeps handling other requests.
    """
    if agent.DEMO_MODE and not _take_demo_slot():
        return ChatResponse(answer=DEMO_LIMIT_MESSAGE.format(limit=DEMO_DAILY_LIMIT))
    return ChatResponse(**asdict(await run_agent(request.question, hub, request.conversation_id, today=_today())))


def _sse(event: dict) -> str:
    """Format one Server-Sent Event: a `data:` line with JSON, ended by a blank line."""
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Same answer as /chat, but streams the agent's progress while it works (Server-Sent Events).

    One question can take 20-60 seconds, so instead of a blank wait the page receives events
    as they happen ("thinking", every tool call as a "step", "retry" on a model fallback) and
    finally one "answer" event shaped like ChatResponse, or an "error" event.

    How: run_agent runs as a background task and pushes each event into a queue through its
    on_event callback; this generator takes them out of the queue and sends them on. If the
    browser disconnects (tab closed), the task is cancelled so it stops spending LLM quota.
    /chat stays as it is: the evals and tests call it.
    """
    if agent.DEMO_MODE and not _take_demo_slot():
        limit_reply = ChatResponse(answer=DEMO_LIMIT_MESSAGE.format(limit=DEMO_DAILY_LIMIT))
        events = iter([_sse({"type": "answer", **limit_reply.model_dump()})])
        return StreamingResponse(events, media_type="text/event-stream")

    async def events():
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        task = asyncio.create_task(
            run_agent(request.question, hub, request.conversation_id, today=_today(), on_event=queue.put)
        )
        task.add_done_callback(lambda _: queue.put_nowait(None))  # None = no more progress events
        try:
            while (event := await queue.get()) is not None:
                yield _sse(event)
            reply = ChatResponse(**asdict(task.result()))
            yield _sse({"type": "answer", **reply.model_dump()})
        except Exception:
            traceback.print_exc()
            yield _sse({"type": "error", "text": "Something went wrong while answering. Please try again."})
        finally:
            task.cancel()  # no-op when finished; stops the agent if the client went away

    # X-Accel-Buffering: stops proxies (like the one in front of Render) from holding events back.
    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/memories")
def memories() -> list[str]:
    """Every long-term fact Mosaic has saved, so the user can see what it remembers.

    Hidden in the public demo: long-term memory is turned off there, and personal facts
    should never be readable by strangers.
    """
    if agent.DEMO_MODE:
        raise HTTPException(status_code=404)
    return memory.all_facts()
