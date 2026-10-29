import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import memory
from agent import ToolHub, run_agent

# Pass the app's MOSAIC_DATA (from the environment, or .env loaded by agent.py) on to the MCP
# servers explicitly: the SDK gives subprocesses only a minimal environment, so a value set on
# the host (e.g. MOSAIC_DATA=mock on Render) would otherwise never reach them.
hub = ToolHub(data_mode=os.getenv("MOSAIC_DATA"))


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
    """Serve the chat page. Serving it from the same server as /chat (same origin) means
    the page's fetch() calls need no CORS setup."""
    return FileResponse(INDEX_PAGE, headers=NO_CACHE)


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
    return ChatResponse(**asdict(await run_agent(request.question, hub, request.conversation_id)))


@app.get("/memories")
def memories() -> list[str]:
    """Every long-term fact Mosaic has saved, so the user can see what it remembers.

    Hidden in the public demo: long-term memory is turned off there, and personal facts
    should never be readable by strangers.
    """
    if agent.DEMO_MODE:
        raise HTTPException(status_code=404)
    return memory.all_facts()
