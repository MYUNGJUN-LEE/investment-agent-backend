from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Callable, TypeVar


DEFAULT_BUSY_TIMEOUT_MS = 30_000
DEFAULT_WRITE_ATTEMPTS = 5

T = TypeVar("T")


class RecoverableSQLiteError(RuntimeError):
    """SQLite error that callers can skip/retry without killing a worker."""


def is_recoverable_sqlite_error(exc: BaseException | str | None) -> bool:
    message = str(exc or "").lower()
    return (
        "database is locked" in message
        or "database table is locked" in message
        or "database schema is locked" in message
        or "locked database" in message
    )


def connect_sqlite(
    path: Path | str,
    *,
    row_factory: bool = False,
    timeout_seconds: float | None = None,
) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path,
        timeout=float(timeout_seconds if timeout_seconds is not None else 30.0),
    )
    if row_factory:
        conn.row_factory = sqlite3.Row
    configure_sqlite_connection(conn)
    return conn


def configure_sqlite_connection(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    conn.execute(f"PRAGMA busy_timeout = {max(1, int(busy_timeout_ms))}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def sqlite_write_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = DEFAULT_WRITE_ATTEMPTS,
    initial_backoff_seconds: float = 0.05,
    max_backoff_seconds: float = 1.0,
) -> T:
    last_error: sqlite3.OperationalError | None = None
    total_attempts = max(1, int(attempts))

    for attempt in range(total_attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if not is_recoverable_sqlite_error(exc):
                raise
            last_error = exc
            if attempt >= total_attempts - 1:
                break
            backoff = min(
                max_backoff_seconds,
                initial_backoff_seconds * (2 ** attempt),
            )
            time.sleep(max(0.0, backoff))

    raise RecoverableSQLiteError(str(last_error or "database is locked"))
