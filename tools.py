"""Tools the agent can call: Gmail search and Notion queries.

These currently read local mock JSON files. Swapping their bodies for real Gmail API and
Notion API calls must keep the function names, parameters and return shapes the same, so the
agent loop never needs to change.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "mock_data"


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

    Raises ValueError for an unknown database name. The agent loop catches this and sends
    the error message back to the LLM, so it can correct itself and call the tool again.
    """
    if database not in NOTION_DATABASES:
        raise ValueError(f"Unknown database '{database}'. Use 'applications' or 'tasks'.")
    rows = NOTION_DATABASES[database]
    if status:
        rows = [row for row in rows if row["status"].lower() == status.lower()]
    return rows


# Tool schemas sent to the LLM. The model never sees the Python code above, only these
# names, descriptions and JSON Schema parameters. It decides which tool to call (and with
# what arguments) purely from this text, so the descriptions matter as much as the code.
TOOL_DECLARATIONS = [
    {
        "name": "search_emails",
        "description": (
            "Search the user's Gmail inbox by keywords. Mails matching more keywords rank "
            "higher; ties are newest first. Results include labels: mails labelled "
            "CATEGORY_PROMOTIONS are usually ads, but can still hold important notices. "
            "Use 1-3 specific keywords per topic (e.g. 'razorpay interview'). To look up "
            "several independent topics (e.g. DBMS, OS lab, ML project), call this tool "
            "several times in the SAME turn, one topic per call, instead of one per turn."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Space-separated keywords."},
                "max_results": {"type": "integer", "description": "Max mails to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_notion",
        "description": (
            "Read the user's Notion databases. 'applications' is the internship tracker "
            "(company, role, status: Applied/Interview/Rejected/Offer, applied date, notes). "
            "'tasks' is the study planner (task, subject, due date, status: Not Started/"
            "In Progress/Done). Notion can be out of date, so cross-check with emails."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "database": {"type": "string", "enum": ["applications", "tasks"]},
                "status": {"type": "string", "description": "Optional status filter, case-insensitive."},
            },
            "required": ["database"],
        },
    },
]

# Maps the tool name the LLM asks for to the Python function the agent loop should run.
TOOL_FUNCTIONS = {
    "search_emails": search_emails,
    "query_notion": query_notion,
}
