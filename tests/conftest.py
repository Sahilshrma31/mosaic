"""Shared test setup, loaded by pytest before any test file.

Forces mock data, so the tests never call the real Gmail or Notion APIs and always see the
same 40 mails with known answers. It must run before `tools` is imported, because tools reads
MOSAIC_DATA at import time; load_dotenv() never overrides a variable that is already set, so
the "live" value in .env doesn't win.
"""

import os

import pytest

os.environ["MOSAIC_DATA"] = "mock"

import memory  # noqa: E402  (imported after MOSAIC_DATA is set on purpose)


@pytest.fixture
def temp_memory(tmp_path, monkeypatch):
    """Point memory.py at an empty SQLite file in a temporary folder, never the real mosaic.db."""
    monkeypatch.setattr(memory, "DB_FILE", tmp_path / "test.db")
    memory.init()
    return memory
