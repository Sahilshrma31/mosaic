"""Tools the agent can call: Gmail search and Notion queries.

These currently read local mock JSON files. Swapping their bodies for real Gmail API and
Notion API calls must keep the function names, parameters and return shapes the same, so the
agent loop never needs to change.
"""

import json
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


def search_emails(query: str, max_results: int = 5) -> list[dict]:
    """Search the inbox and return the best-matching emails, newest first.

    Matching is a simple keyword AND search, like Gmail's own search box: every word in
    `query` must appear somewhere in the sender, subject or body (case-insensitive).
    For example, "razorpay interview" matches only mails containing both words.

    Results are sorted newest first on purpose. When two mails disagree (a deadline of
    25 Oct in one mail, extended to 30 Oct in a later one), the latest information comes
    first, which helps the LLM pick the correct answer.

    Returns at most `max_results` full email records (id, from, subject, date, labels,
    body, url). An empty list means nothing matched, and the LLM can retry with fewer
    or different keywords.
    """
    words = query.lower().split()
    matches = []
    for mail in EMAILS:
        text = f"{mail['from']} {mail['subject']} {mail['body']}".lower()
        if all(word in text for word in words):
            matches.append(mail)
    matches.sort(key=lambda mail: mail["date"], reverse=True)
    return matches[:max_results]


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
            "Search the user's Gmail inbox by keywords. Every keyword must appear in the "
            "sender, subject or body. Results are newest first and include labels: mails "
            "labelled CATEGORY_PROMOTIONS are usually ads, but can still hold important "
            "notices. Use few, specific keywords (e.g. 'razorpay interview'); if nothing "
            "matches, retry with fewer or different keywords."
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
