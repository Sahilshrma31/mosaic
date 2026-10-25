"""Seed the test Gmail inbox with the mock emails, without sending anything.

users.messages.insert drops a message straight into the mailbox, like a delivered mail, so
the From address, date and labels can all match mock_data/emails.json. The planted tricky
cases (extended deadline, Swiggy rejection, the Google test mail in Promotions) therefore
behave the same in real Gmail as in the mock data.
"""

import base64
import json
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from gmail_auth import get_gmail_service

EMAILS_FILE = Path(__file__).parent / "mock_data" / "emails.json"


def build_raw_message(mail: dict, to_address: str) -> str:
    """Turn one mock email into a real RFC 2822 email, base64url-encoded as Gmail expects.

    The Message-ID is derived from the mock id (msg_001 -> <msg_001@mosaic.test>), so each
    mail can be found again later with the search `rfc822msgid:msg_001@mosaic.test`.
    """
    message = EmailMessage()
    message["From"] = mail["from"]
    message["To"] = to_address
    message["Subject"] = mail["subject"]
    message["Date"] = format_datetime(datetime.fromisoformat(mail["date"]))
    message["Message-ID"] = f"<{mail['id']}@mosaic.test>"
    message.set_content(mail["body"])
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


def already_seeded(service, mail_id: str) -> bool:
    """True if a mail with this mock id is already in the mailbox (keeps the script idempotent)."""
    found = service.users().messages().list(userId="me", q=f"rfc822msgid:{mail_id}@mosaic.test").execute()
    return bool(found.get("messages"))


def main() -> None:
    service = get_gmail_service()
    to_address = service.users().getProfile(userId="me").execute()["emailAddress"]
    emails = json.loads(EMAILS_FILE.read_text())

    inserted = skipped = 0
    for mail in emails:
        if already_seeded(service, mail["id"]):
            skipped += 1
            continue
        body = {"raw": build_raw_message(mail, to_address), "labelIds": mail["labels"]}
        # dateHeader: Gmail sorts and shows the mail by its Date header (the mock date),
        # not by the moment we inserted it.
        service.users().messages().insert(userId="me", body=body, internalDateSource="dateHeader").execute()
        inserted += 1
        print(f"inserted {mail['id']}: {mail['subject'][:60]}")

    print(f"\nDone: {inserted} inserted, {skipped} already there.")


if __name__ == "__main__":
    main()
