"""One-time setup: create the Applications and Study Planner databases in Notion and fill them
with the mock rows, under the page whose id is NOTION_PARENT_PAGE_ID in .env.

Idempotent: the new database ids are saved to .env, and if they're already there the script
stops, so running it twice never creates duplicate databases.
"""

import json
import os
import time
from pathlib import Path

from dotenv import set_key

import notion_api

MOCK_FILES = {"applications": "notion_applications.json", "tasks": "notion_tasks.json"}


def main() -> None:
    env_vars = [spec["env_var"] for spec in notion_api.DATABASES.values()]
    if any(os.getenv(var) for var in env_vars):
        print(f"Already seeded ({', '.join(env_vars)} found in .env). Nothing to do.")
        return

    parent_page_id = os.environ["NOTION_PARENT_PAGE_ID"]
    for key, filename in MOCK_FILES.items():
        database_id = notion_api.create_database(key, parent_page_id)
        print(f"Created '{notion_api.DATABASES[key]['title']}' database")

        rows = json.loads((Path(__file__).parent / "mock_data" / filename).read_text())
        for row in rows:
            notion_api.add_row(key, database_id, row)
            time.sleep(0.35)  # stay under Notion's ~3 requests/second limit
        print(f"  added {len(rows)} rows")

        # Saved right after each database, so a crash halfway still leaves .env accurate.
        set_key(notion_api.ENV_FILE, notion_api.DATABASES[key]["env_var"], database_id)

    print("Done. Database ids saved to .env.")


if __name__ == "__main__":
    main()
