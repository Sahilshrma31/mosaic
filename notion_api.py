"""Minimal Notion REST client: create the two databases, add rows, and read rows back as flat dicts.

Talks to the Notion API directly with httpx. One table (DATABASES) describes both databases,
and it drives everything: the columns created by the seed script, the properties written for
each row, and how Notion's nested JSON is flattened back into the same dicts the mock returns.
"""

import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Loaded by path, not from the environment: MCP servers run as subprocesses and the MCP SDK
# passes them only a minimal set of environment variables, so they read .env themselves.
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

BASE_URL = "https://api.notion.com/v1"
# Pinned on purpose: newer Notion API versions restructure databases, and a pinned version
# keeps this code working the same way until we choose to upgrade.
NOTION_VERSION = "2022-06-28"

# field in our dicts, Notion column name, Notion property type
DATABASES = {
    "applications": {
        "title": "Applications",
        "env_var": "NOTION_APPLICATIONS_DB_ID",
        "columns": [
            ("company", "Company", "title"),
            ("role", "Role", "rich_text"),
            ("status", "Status", "select"),
            ("applied_date", "Applied Date", "date"),
            ("notes", "Notes", "rich_text"),
        ],
        "options": {"Status": [("Applied", "blue"), ("Interview", "yellow"), ("Rejected", "red"), ("Offer", "green")]},
    },
    "tasks": {
        "title": "Study Planner",
        "env_var": "NOTION_TASKS_DB_ID",
        "columns": [
            ("task", "Task", "title"),
            ("subject", "Subject", "select"),
            ("due_date", "Due Date", "date"),
            ("status", "Status", "select"),
        ],
        "options": {"Status": [("Not Started", "gray"), ("In Progress", "yellow"), ("Done", "green")]},
    },
}


def _request(method: str, path: str, body: dict | None = None) -> dict:
    """Call one Notion endpoint and return the JSON reply.

    Notion allows about 3 requests per second and answers 429 when that's exceeded, with a
    Retry-After header saying how long to wait; we wait and retry up to 3 times. Any other
    error is raised with Notion's own message (e.g. "Could not find database"), which is far
    more useful than a bare status code when it reaches the LLM or the terminal.
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
        "Notion-Version": NOTION_VERSION,
    }
    for _ in range(3):
        response = httpx.request(method, f"{BASE_URL}{path}", headers=headers, json=body, timeout=30)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", "1")))
            continue
        if response.is_error:
            data = response.json()
            raise RuntimeError(f"Notion API {response.status_code} {data.get('code')}: {data.get('message')}")
        return response.json()
    raise RuntimeError("Notion API rate limit: still 429 after 3 retries")


def _column_schema(notion_type: str, options: list[tuple[str, str]] | None) -> dict:
    """The column definition Notion expects when creating a database, e.g. {"date": {}}."""
    if notion_type == "select":
        return {"select": {"options": [{"name": name, "color": color} for name, color in options or []]}}
    return {notion_type: {}}


def _to_property(notion_type: str, value: str | None) -> dict:
    """Wrap one plain value in the nested shape Notion stores it in.

    For example "Swiggy" as a title becomes {"title": [{"text": {"content": "Swiggy"}}]}.
    Empty values become an empty property instead of an error.
    """
    if notion_type in ("title", "rich_text"):
        return {notion_type: [{"text": {"content": value}}] if value else []}
    if notion_type == "select":
        return {"select": {"name": value} if value else None}
    if notion_type == "date":
        return {"date": {"start": value} if value else None}
    raise ValueError(f"Unsupported Notion property type: {notion_type}")


def _from_property(prop: dict) -> str | None:
    """Unwrap one Notion property back into a plain value (the reverse of _to_property)."""
    kind = prop["type"]
    if kind in ("title", "rich_text"):
        return "".join(part["plain_text"] for part in prop[kind])
    if kind == "select":
        return prop["select"]["name"] if prop["select"] else None
    if kind == "date":
        return prop["date"]["start"] if prop["date"] else None
    return None


def create_database(key: str, parent_page_id: str) -> str:
    """Create one database (columns from DATABASES) inside the parent page and return its id."""
    spec = DATABASES[key]
    properties = {
        name: _column_schema(notion_type, spec["options"].get(name))
        for _, name, notion_type in spec["columns"]
    }
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": spec["title"]}}],
        "properties": properties,
    }
    return _request("POST", "/databases", body)["id"]


def add_row(key: str, database_id: str, row: dict) -> None:
    """Add one row (a dict shaped like the mock JSON) to the database as a Notion page."""
    properties = {
        name: _to_property(notion_type, row.get(field))
        for field, name, notion_type in DATABASES[key]["columns"]
    }
    _request("POST", "/pages", {"parent": {"database_id": database_id}, "properties": properties})


def query(key: str) -> list[dict]:
    """Return every row of one database as flat dicts, in the order they were created.

    Notion returns at most 100 rows per request, with `has_more` and `next_cursor` for the
    rest, so we keep asking until has_more is false. Each Notion page is flattened into the
    same keys the mock data uses, plus the real page id and url (used later for citations).
    """
    database_id = os.getenv(DATABASES[key]["env_var"])
    if not database_id:
        raise RuntimeError(f"{DATABASES[key]['env_var']} is not set. Run seed_notion.py first.")

    pages, cursor = [], None
    while True:
        body = {"page_size": 100, "sorts": [{"timestamp": "created_time", "direction": "ascending"}]}
        if cursor:
            body["start_cursor"] = cursor
        data = _request("POST", f"/databases/{database_id}/query", body)
        pages.extend(data["results"])
        if not data["has_more"]:
            break
        cursor = data["next_cursor"]

    rows = []
    for page in pages:
        row = {"id": page["id"]}
        for field, name, _ in DATABASES[key]["columns"]:
            row[field] = _from_property(page["properties"][name])
        row["url"] = page["url"]
        rows.append(row)
    return rows
