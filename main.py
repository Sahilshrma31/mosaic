from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import ToolHub, run_agent

hub = ToolHub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the MCP servers once when the app boots, and shut them down when it stops.

    Code before `yield` runs at startup, code after it at shutdown. Every request in
    between reuses the same open connections instead of launching new server processes.
    """
    await hub.start()
    yield
    await hub.stop()


app = FastAPI(title="Mosaic", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_PAGE = STATIC_DIR / "index.html"

# Serves the compiled frontend (chat.js, built from frontend/chat.ts with `npm run build`).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Serve the chat page. Serving it from the same server as /chat (same origin) means
    the page's fetch() calls need no CORS setup."""
    return FileResponse(INDEX_PAGE)


class ChatRequest(BaseModel):
    """Request body for POST /chat, e.g. {"question": "When is my Razorpay interview?"}.

    FastAPI validates incoming JSON against this model and rejects a body without a
    `question` string with a 422 error, before any LLM call is made.
    """

    question: str


class ChatResponse(BaseModel):
    """Response body for POST /chat, e.g. {"answer": "Your interview is on 28 Oct..."}."""

    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer the user's question with the agent loop over their Gmail and Notion data.

    This route only handles HTTP: validating the request and shaping the response.
    All agent logic lives in agent.py, so the agent can change without touching this file.
    It's `async def` because run_agent awaits the LLM and the MCP servers; while one
    question waits on them, the server keeps handling other requests.
    """
    return ChatResponse(answer=await run_agent(request.question, hub))
