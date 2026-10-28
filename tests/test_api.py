"""Tests for the HTTP layer in main.py. The agent is replaced by a fake, so no Gemini call is made.

TestClient is created without `with`, so FastAPI's lifespan doesn't run and no MCP servers
are started; these tests only need the routes.
"""

from fastapi.testclient import TestClient

import main
from agent import AgentReply

client = TestClient(main.app)


def test_chat_rejects_missing_question():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_rejects_too_long_conversation_id():
    response = client.post("/chat", json={"question": "Hi", "conversation_id": "x" * 65})
    assert response.status_code == 422


def test_chat_returns_answer_sources_and_trace(monkeypatch):
    async def fake_run_agent(question, hub, conversation_id=None):
        return AgentReply(
            answer="Swiggy rejected you [1].",
            sources=[{"n": 1, "source": "gmail", "title": "Swiggy rejection", "url": "u1", "date": "2026-10-20"}],
            trace=[{"round": 1, "call": "search_emails(query='swiggy')", "result": "2 results"}],
        )

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    response = client.post("/chat", json={"question": "What happened with Swiggy?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Swiggy rejected you [1]."
    assert data["sources"][0]["title"] == "Swiggy rejection"
    assert data["trace"][0]["call"] == "search_emails(query='swiggy')"
    assert data["plan"] == []


def test_memories_lists_saved_facts(temp_memory):
    temp_memory.remember("Prefers backend roles")
    response = client.get("/memories")
    assert response.json() == ["Prefers backend roles"]
