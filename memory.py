"""Mosaic's memory, stored in a local SQLite file (mosaic.db).

Two kinds:
- Short-term: the messages of one conversation (table `messages`, keyed by conversation_id),
  so a follow-up like "What about Round 2?" knows what was asked before.
- Long-term: lasting facts about the student, e.g. "Google is the student's dream company"
  (table `memories`), shared by every conversation and added to the system prompt.

Also holds proposed Notion updates waiting for the student's approval (table `pending_actions`).

Each function opens its own short connection: SQLite is a single local file, connecting is
cheap, and it keeps the functions safe to call from any request.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_FILE = Path(__file__).parent / "mosaic.db"

# How many past question/answer pairs a new question sees. Older turns are left out to keep
# the prompt small: every extra token is resent on every round of the agent loop.
HISTORY_TURNS = 6

# How long a proposed Notion update can still be approved.
ACTION_TTL = timedelta(hours=1)


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init() -> None:
    """Create the tables if they don't exist yet. Safe to call on every startup."""
    with _connect() as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id)")
        db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS pending_actions (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                row_id TEXT NOT NULL,
                title TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def add_message(conversation_id: str, role: str, content: str) -> None:
    """Save one turn of a conversation ("user" question or "assistant" answer)."""
    with _connect() as db:
        db.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, _now()),
        )


def recent_history(conversation_id: str) -> list[tuple[str, str]]:
    """The last HISTORY_TURNS question/answer pairs of a conversation, oldest first.

    Fetched newest-first with a LIMIT (so only the tail is read), then reversed into
    reading order for the prompt.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, HISTORY_TURNS * 2),
        ).fetchall()
    return list(reversed(rows))


def remember(fact: str) -> bool:
    """Save a long-term fact. Returns False if the exact fact was already saved (UNIQUE column)."""
    with _connect() as db:
        cursor = db.execute("INSERT OR IGNORE INTO memories (fact, created_at) VALUES (?, ?)", (fact.strip(), _now()))
        return cursor.rowcount == 1


def all_facts() -> list[str]:
    """Every long-term fact, oldest first."""
    with _connect() as db:
        return [row[0] for row in db.execute("SELECT fact FROM memories ORDER BY id")]


def create_action(conversation_id: str, row_id: str, title: str, from_status: str, to_status: str) -> str:
    """Store an update the agent proposed, waiting for the student's decision. Returns its id.

    The server keeps what will change, so an approval click can only confirm this exact
    update: the browser sends just the id, never a row or a status of its own.
    """
    action_id = uuid.uuid4().hex
    with _connect() as db:
        db.execute(
            "INSERT INTO pending_actions (id, conversation_id, row_id, title, from_status, to_status, state, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (action_id, conversation_id, row_id, title, from_status, to_status, _now()),
        )
    return action_id


def claim_action(action_id: str, conversation_id: str) -> dict | None:
    """Take a pending action for processing, or return None if it can't be decided any more.

    One UPDATE both checks and changes the state, so a double click can't run an update twice.
    It must belong to the same conversation and be under ACTION_TTL old: an old card shouldn't
    apply a change based on data that may no longer be true.
    """
    cutoff = (datetime.now(timezone.utc) - ACTION_TTL).isoformat()
    with _connect() as db:
        claimed = db.execute(
            "UPDATE pending_actions SET state = 'processing'"
            " WHERE id = ? AND conversation_id = ? AND state = 'pending' AND created_at >= ?",
            (action_id, conversation_id, cutoff),
        ).rowcount
        if not claimed:
            return None
        row = db.execute(
            "SELECT id, row_id, title, from_status, to_status FROM pending_actions WHERE id = ?", (action_id,)
        ).fetchone()
    return dict(zip(("id", "row_id", "title", "from_status", "to_status"), row))


def finish_action(action_id: str, state: str) -> None:
    """Record how a claimed action ended: applied, rejected, stale or failed."""
    with _connect() as db:
        db.execute("UPDATE pending_actions SET state = ? WHERE id = ?", (state, action_id))
