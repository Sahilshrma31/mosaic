"""Tests for the eval harness itself: the grader, the case file, and case isolation. No LLM calls.

A broken case (a regex that can never match, a source id that doesn't exist in the mock data)
would fail on every run and silently lower the accuracy numbers, so the cases are checked too.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

import agent
from evals.run_eval import CASES_FILE, run_case, score

MOCK_DIR = Path(__file__).resolve().parent.parent / "mock_data"
MOCK_IDS = {
    row["id"]
    for name in ("emails.json", "notion_applications.json", "notion_tasks.json")
    for row in json.loads((MOCK_DIR / name).read_text())
}
CASES = json.loads(CASES_FILE.read_text())

DBMS = {
    "must_include": ["30(st|nd|rd|th)?\\s*oct|oct\\w*\\s*30|10-30"],
    "expected_sources": [["msg_002"]],
    "relevant_sources": ["msg_001", "msg_002", "task_001"],
}


@pytest.mark.parametrize("answer", ["Due **30 Oct**.", "Due 30th October.", "Due October 30.", "Due 2026-10-30."])
def test_date_pattern_accepts_wording_variants(answer):
    assert score(DBMS, answer, ["msg_002"])["answer_pass"]


def test_old_deadline_fails_the_answer():
    result = score(DBMS, "Your DBMS assignment is due on 25 Oct.", ["msg_001"])
    assert not result["answer_pass"] and result["missing"]


def test_missing_expected_source_fails_citations():
    result = score(DBMS, "Due 30 Oct.", ["task_001"])
    assert result["citation_pass"] is False and result["uncited"] == [["msg_002"]]


def test_unrelated_citation_fails_citations():
    result = score(DBMS, "Due 30 Oct.", ["msg_002", "msg_025"])
    assert result["citation_pass"] is False and result["irrelevant"] == ["msg_025"]


def test_forbidden_pattern_fails_the_answer():
    case = {"must_include": ["not"], "must_not_include": ["amazon[^.\\n]{0,60}\\b\\d{1,2}\\s*oct"]}
    assert score(case, "I could not find it.", [])["answer_pass"]
    assert not score(case, "Not sure, but your Amazon interview is on 29 Oct.", [])["answer_pass"]


def test_unanswerable_case_is_not_scored_for_citations():
    case = {"must_include": ["not"], "expected_sources": [], "relevant_sources": None}
    assert score(case, "I could not find it.", [])["citation_pass"] is None


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_is_well_formed(case):
    for pattern in case.get("must_include", []) + case.get("must_not_include", []):
        re.compile(pattern)
    expected_ids = {i for group in case.get("expected_sources", []) for i in group}
    relevant = case.get("relevant_sources")
    assert expected_ids <= MOCK_IDS, "expected source not in the mock data"
    if relevant is not None:
        assert set(relevant) <= MOCK_IDS, "relevant source not in the mock data"
        assert expected_ids <= set(relevant), "an expected source must also count as relevant"


def test_case_ids_are_unique():
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))


def test_run_case_isolates_memory_and_pins_today(monkeypatch):
    seen = {}

    async def fake_run_agent(question, hub, conversation_id, today):
        import memory

        seen["facts"] = memory.all_facts()
        seen["history"] = memory.recent_history(conversation_id)
        seen["today"] = today.isoformat()
        return agent.AgentReply("Round 2 is a low-level design round [1].",
                                sources=[{"n": 1, "id": "msg_004"}], model="fake-model")

    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    case = {
        "id": "t", "question": "What about Round 2?", "today": "2026-10-26",
        "facts": ["The student prefers backend roles."],
        "history": [["user", "When is my Razorpay interview?"], ["assistant", "28 Oct."]],
        "must_include": ["low.?level design"], "expected_sources": [["msg_004"]], "relevant_sources": ["msg_004"],
    }
    result = asyncio.run(run_case(case, hub=None))

    assert seen == {
        "facts": ["The student prefers backend roles."],
        "history": [("user", "When is my Razorpay interview?"), ("assistant", "28 Oct.")],
        "today": "2026-10-26",
    }
    assert result["answer_pass"] and result["citation_pass"] and result["model"] == "fake-model"
