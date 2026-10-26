from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import memory
from agent import ToolHub, run_agent

hub = ToolHub()


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

    question: str
    conversation_id: str | None = Field(default=None, max_length=64)


class Source(BaseModel):
    """One cited source. `n` matches the [n] in the answer text."""

    n: int
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer the user's question with the agent loop over their Gmail and Notion data.

    This route only handles HTTP: validating the request and shaping the response.
    All agent logic lives in agent.py, so the agent can change without touching this file.
    It's `async def` because run_agent awaits the LLM and the MCP servers; while one
    question waits on them, the server keeps handling other requests.
    """
    return ChatResponse(**asdict(await run_agent(request.question, hub, request.conversation_id)))


@app.get("/memories")
def memories() -> list[str]:
    """Every long-term fact Mosaic has saved, so the user can see what it remembers."""
    return memory.all_facts()
