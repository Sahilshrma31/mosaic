"""Data access behind the MCP servers: Gmail search and Notion queries.

gmail_server.py and notion_server.py expose these functions as MCP tools. MOSAIC_DATA in
.env picks the backend: "mock" (default) reads the local JSON files, whose correct answers
are known, which the evals rely on; "live" calls the real APIs. Both backends return the
same shapes, so neither the servers nor the agent know which one is in use.
"""

import json
import os
import re
from pathlib import Path

import notion_api  # also loads .env

DATA_DIR = Path(__file__).parent / "mock_data"
LIVE = os.getenv("MOSAIC_DATA", "mock") == "live"


def _load(filename: str) -> list[dict]:
    """Read one mock JSON file from the mock_data folder and return its list of records."""
    with open(DATA_DIR / filename) as f:
        return json.load(f)


# Loaded once at import time, not on every tool call, so repeated searches stay fast.
EMAILS = _load("emails.json")
NOTION_DATABASES = {
    "applications": _load("notion_applications.json"),
    "tasks": _load("notion_tasks.json"),
}


def _words(text: str) -> set[str]:
    """Split text into lowercase whole words, so "os" matches "OS" but not "most" or "post"."""
    return set(re.findall(r"\w+", text.lower()))


def search_emails(query: str, max_results: int = 5) -> list[dict]:
    """Search the inbox and return the best-matching emails, ranked by relevance.

    A mail matches if it contains at least one query word (whole words, case-insensitive)
    in its sender, subject or body. Mails matching more of the query words rank higher, so
    "swiggy application" puts the rejection mail (both words) above a Swiggy food promo
    (one word). Requiring every word instead made multi-word queries like
    "assignment submission deadline" return nothing at all.

    Ties are broken newest first on purpose. When two mails disagree (a deadline of
    25 Oct in one mail, extended to 30 Oct in a later one), the latest information comes
    first, which helps the LLM pick the correct answer.

    Returns at most `max_results` full email records (id, from, subject, date, labels,
    body, url). An empty list means no word matched, and the LLM can retry with
    different keywords.
    """
    query_words = _words(query)
    scored = []
    for mail in EMAILS:
        mail_words = _words(f"{mail['from']} {mail['subject']} {mail['body']}")
        score = len(query_words & mail_words)
        if score > 0:
            scored.append((score, mail["date"], mail))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [mail for _, _, mail in scored[:max_results]]


def query_notion(database: str, status: str | None = None) -> list[dict]:
    """Return rows from one Notion database, optionally filtered by status.

    `database` must be "applications" (the internship tracker: company, role, status,
    applied date, notes) or "tasks" (the study planner: task, subject, due date, status).
    `status` is matched case-insensitively, e.g. "applied", "interview" or "not started".
    With no status, every row of the database is returned.

    In live mode the rows come from the Notion API on every call, so edits made in Notion
    show up immediately. Status is filtered here in Python rather than in the Notion query,
    because Notion's filters are case-sensitive and the LLM may send "applied" or "Applied".

    Raises ValueError for an unknown database name, and RuntimeError if the Notion API
    fails. The agent loop sends either error message back to the LLM instead of crashing.
    """
    if database not in NOTION_DATABASES:
        raise ValueError(f"Unknown database '{database}'. Use 'applications' or 'tasks'.")
    rows = notion_api.query(database) if LIVE else NOTION_DATABASES[database]
    if status:
        rows = [row for row in rows if row["status"].lower() == status.lower()]
    return rows
