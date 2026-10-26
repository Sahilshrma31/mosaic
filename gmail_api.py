"""Live Gmail search: fetch candidate mails from the Gmail API and rank them like the mock does.

Gmail's own search ANDs the keywords ("DBMS OS ML project" only matches mails containing all
four words), which made multi-word queries return nothing. So we ask Gmail for mails matching
ANY of the words, flatten each one into the same dict the mock data uses, and let tools.rank()
pick the best ones with the exact scoring the mock search uses.
"""

import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from gmail_auth import get_gmail_service
from tools import rank, words

# How many matching mails to fetch before ranking. Gmail returns matches newest first, not by
# relevance, so this must be comfortably larger than max_results. They're fetched in one batch
# request, so a larger number costs response size rather than extra round trips.
CANDIDATES = 15

_service = None
_account = None  # the logged-in address, e.g. mosaic.test.sahil@gmail.com


def _gmail():
    """Build the Gmail client once per process and reuse it. It refreshes its own access token."""
    global _service, _account
    if _service is None:
        _service = get_gmail_service(interactive=False)
        _account = _service.users().getProfile(userId="me").execute()["emailAddress"]
    return _service


def _web_url(thread_id: str) -> str:
    """Link that opens this mail in Gmail on the right account.

    "/mail/u/0/" means "the first account signed in to this browser", which is usually the
    user's main Gmail, not the test account. `authuser=<address>` makes Gmail switch to the
    account the mail actually belongs to. The web UI opens conversations by thread id.
    """
    return f"https://mail.google.com/mail/?authuser={quote(_account or '', safe='@')}#all/{thread_id}"


def _body_text(payload: dict) -> str:
    """Find the text/plain part of a message and decode it.

    Gmail returns the body base64url-encoded, and multipart mails nest it inside `parts`
    (possibly several levels deep), so we search the tree for the first text/plain part.
    """
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace").strip()
    for part in payload.get("parts", []):
        text = _body_text(part)
        if text:
            return text
    return ""


def _flatten(message: dict) -> dict:
    """Turn one Gmail API message into the flat dict shape of mock_data/emails.json."""
    headers = {h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])}
    try:
        date = parsedate_to_datetime(headers["date"]).isoformat()
    except (KeyError, TypeError, ValueError):
        # No usable Date header: fall back to when Gmail received the mail (ms since epoch).
        date = datetime.fromtimestamp(int(message["internalDate"]) / 1000, tz=timezone.utc).isoformat()
    return {
        "id": message["id"],
        "thread_id": message["threadId"],
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": date,
        "labels": message.get("labelIds", []),
        "body": _body_text(message["payload"]) or message.get("snippet", ""),
        "url": _web_url(message["threadId"]),
    }


def _fetch_all(gmail, message_ids: list[str]) -> list[dict]:
    """Fetch full messages for all ids in ONE HTTP round trip using a Gmail batch request.

    Calling messages.get once per id took ~0.8s each on a slow network (7s for 8 mails).
    A batch packs all the gets into a single request and Google returns every reply together.
    Each reply arrives through the callback; a mail that fails on its own (e.g. deleted in
    the meantime) is skipped, but if every one fails, the first error is raised.
    """
    messages, errors = [], []

    def collect(request_id, response, exception):
        if exception is None:
            messages.append(response)
        else:
            errors.append(exception)

    batch = gmail.new_batch_http_request(callback=collect)
    for message_id in message_ids:
        batch.add(gmail.users().messages().get(userId="me", id=message_id, format="full"))
    batch.execute()

    if errors and not messages:
        raise errors[0]
    return messages


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return the best-matching mails from the real inbox, ranked by tools.rank().

    Two Gmail calls in total: messages.list with an OR query to get candidate ids, then one
    batch request that fetches all those messages.
    """
    query_words = sorted(words(query))
    if not query_words:
        return []

    gmail = _gmail()
    found = gmail.users().messages().list(userId="me", q=" OR ".join(query_words), maxResults=CANDIDATES).execute()
    message_ids = [ref["id"] for ref in found.get("messages", [])]
    if not message_ids:
        return []
    mails = [_flatten(message) for message in _fetch_all(gmail, message_ids)]
    return rank(query, mails, max_results)
