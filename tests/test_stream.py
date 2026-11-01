"""Tests for live progress: agent events (run_agent's on_event) and the /chat/stream endpoint.

No Gemini call is made: the endpoint tests replace run_agent with a fake, and the retry test
replaces the inner loop with one that fails with a server error.
"""

import asyncio
import json

from fastapi.testclient import TestClient
from google.genai import errors

import agent
import main
from agent import AgentReply

client = TestClient(main.app)


def stream_events(question: str) -> list[dict]:
    """POST to /chat/stream and decode every `data: {...}` event in the reply, in order."""
    response = client.post("/chat/stream", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(block[len("data: "):]) for block in response.text.split("\n\n") if block.startswith("data: ")]


def test_stream_sends_progress_then_answer(monkeypatch):
    async def fake_run_agent(question, hub, conversation_id=None, today=None, on_event=None):
        await on_event({"type": "thinking", "round": 1})
        await on_event({"type": "step", "round": 1, "tool": "search_emails", "args": {"query": "swiggy"}, "result": "2 results"})
        return AgentReply(answer="Swiggy rejected you [1].",
                          sources=[{"n": 1, "source": "gmail", "title": "Swiggy rejection", "url": "u1", "date": "2026-10-20"}])

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    events = stream_events("What happened with Swiggy?")

    assert [event["type"] for event in events] == ["thinking", "step", "answer"]
    assert events[1]["args"] == {"query": "swiggy"}
    assert events[-1]["answer"] == "Swiggy rejected you [1]."
    assert events[-1]["sources"][0]["title"] == "Swiggy rejection"


def test_stream_turns_a_crash_into_an_error_event(monkeypatch):
    async def broken_run_agent(question, hub, conversation_id=None, today=None, on_event=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_agent", broken_run_agent)
    events = stream_events("Anything")

    assert events[-1]["type"] == "error"
    assert "boom" not in events[-1]["text"]  # internals stay in the server log, not on the page


def test_model_fallback_emits_retry(monkeypatch):
    """A 503 on the first model sends a "retry" event before the next model starts over."""
    attempts = []

    async def fake_loop(question, model, hub, history, today=None, on_event=None):
        attempts.append(model)
        if model == "first-model":
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        return AgentReply(answer="ok", model=model)

    monkeypatch.setattr(agent, "MODELS", ["first-model", "second-model"])
    monkeypatch.setattr(agent, "_run_loop", fake_loop)
    events: list[dict] = []

    async def collect(event: dict) -> None:
        events.append(event)

    reply = asyncio.run(agent.run_agent("Hi", hub=None, on_event=collect))

    assert attempts == ["first-model", "second-model"]
    assert events == [{"type": "retry"}]
    assert reply.answer == "ok"
