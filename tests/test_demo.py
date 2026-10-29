"""Tests for the public demo mode (MOSAIC_DEMO=1). No Gemini calls: the agent is faked.

The demo is used by strangers, so it must cap questions per day (free AI quota), keep
long-term memory read- and write-protected, and reject oversized questions.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent
import main
from agent import AgentReply

client = TestClient(main.app)


@pytest.fixture
def demo(monkeypatch):
    """Switch demo mode on with a limit of 2 questions and a fresh counter."""
    monkeypatch.setattr(agent, "DEMO_MODE", True)
    monkeypatch.setattr(main, "DEMO_DAILY_LIMIT", 2)
    monkeypatch.setattr(main, "_demo_usage", {"day": "", "count": 0})


def test_demo_stops_answering_after_the_daily_limit(demo, monkeypatch):
    calls = []

    async def fake_run_agent(question, hub, conversation_id=None):
        calls.append(question)
        return AgentReply(answer="An answer.")

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    answers = [client.post("/chat", json={"question": f"q{i}"}).json()["answer"] for i in range(3)]

    assert answers[:2] == ["An answer.", "An answer."]
    assert "today" in answers[2] and "2 questions" in answers[2]
    assert calls == ["q0", "q1"]  # the third question never reached the agent (no quota spent)


def test_demo_hides_long_term_memory(demo):
    assert client.get("/memories").status_code == 404


def test_memories_are_visible_outside_the_demo(temp_memory):
    assert client.get("/memories").status_code == 200


def test_oversized_question_is_rejected():
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422


def test_demo_does_not_offer_the_remember_tool(demo, temp_memory):
    config, _ = agent._configs(SimpleNamespace(declarations=[]))
    names = [declaration.name for declaration in config.tools[0].function_declarations]
    assert names == ["plan"]


def test_remember_tool_is_offered_outside_the_demo(temp_memory):
    config, _ = agent._configs(SimpleNamespace(declarations=[]))
    names = [declaration.name for declaration in config.tools[0].function_declarations]
    assert names == ["plan", "remember"]
