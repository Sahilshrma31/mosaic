"""Gmail OAuth: log in once in the browser and save the tokens to token.json.

Other scripts (the Gmail seed script, the Gmail MCP server) call get_gmail_service() to get
an authorized Gmail API client. The first run opens a browser for the Google login; later
runs reuse token.json and refresh the short-lived access token automatically.
"""

from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"  # the app's identity (OAuth client)
TOKEN_FILE = BASE_DIR / "token.json"  # the test user's permission (access + refresh token)

# Least privilege: read mail, and insert mail for the seed script. No send, no delete.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.insert",
]


def get_gmail_service(interactive: bool = True):
    """Return an authorized Gmail API client, opening the browser login only when needed.

    The access token lasts about an hour, so an expired one is refreshed silently with the
    refresh token. In Testing mode the refresh token itself expires after 7 days (or the user
    can revoke it); then refreshing fails and we fall back to a fresh browser login.

    With interactive=False (used by the Gmail MCP server, a background subprocess with no one
    watching) there is no browser fallback: it raises a clear error instead of hanging while
    it waits for a login nobody will see.
    """
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            creds = None

    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError("Gmail login missing or expired. Run `python gmail_auth.py` to log in again.")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)  # opens the browser and waits for "Allow"

    TOKEN_FILE.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


if __name__ == "__main__":
    profile = get_gmail_service().users().getProfile(userId="me").execute()
    print("Logged in as:", profile["emailAddress"])
    print("Messages in mailbox:", profile["messagesTotal"])
