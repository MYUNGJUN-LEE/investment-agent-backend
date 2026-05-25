from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from app.config import settings


@contextmanager
def sqlite_connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection.

    This small layer keeps current SQLite usage intact and gives us one place to
    replace with Postgres/Supabase later.
    """
    conn = sqlite3.connect(settings.storage_path(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
