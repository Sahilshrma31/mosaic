"""Tests for the HTTP layer in main.py. The agent is replaced by a fake, so no Gemini call is made.

TestClient is created without `with`, so FastAPI's lifespan doesn't run and no MCP servers
are started; these tests only need the routes.
"""

from datetime import date

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
    async def fake_run_agent(question, hub, conversation_id=None, today=None):
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


def test_chat_page_is_served():
    response = client.get("/chat")
    assert response.status_code == 200
    assert 'id="messages"' in response.text


def capture_today(monkeypatch) -> list:
    """Replace the agent with a fake that records the `today` it was given."""
    seen = []

    async def fake_run_agent(question, hub, conversation_id=None, today=None):
        seen.append(today)
        return AgentReply(answer="ok")

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    return seen


def test_mock_data_pins_today_to_the_mock_date(monkeypatch):
    """On mock data the agent must think it's 24 Oct 2026, or every planted deadline is past."""
    monkeypatch.setattr(main, "DATA_MODE", "mock")
    monkeypatch.delenv("MOSAIC_TODAY", raising=False)
    seen = capture_today(monkeypatch)
    client.post("/chat", json={"question": "What's due today?"})
    assert seen == [date(2026, 10, 24)]
    assert client.get("/config").json()["today"] == "2026-10-24"


def test_live_data_uses_the_real_date(monkeypatch):
    monkeypatch.setattr(main, "DATA_MODE", "live")
    monkeypatch.delenv("MOSAIC_TODAY", raising=False)
    seen = capture_today(monkeypatch)
    client.post("/chat", json={"question": "What's due today?"})
    assert seen == [None]  # None = run_agent uses the real date
    assert client.get("/config").json() == {"data": "live", "demo": main.agent.DEMO_MODE, "today": None}


def test_mosaic_today_overrides_the_date(monkeypatch):
    monkeypatch.setenv("MOSAIC_TODAY", "2026-10-26")
    seen = capture_today(monkeypatch)
    client.post("/chat", json={"question": "What's due today?"})
    assert seen == [date(2026, 10, 26)]
