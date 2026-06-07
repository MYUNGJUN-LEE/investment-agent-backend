from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any
from uuid import uuid4

from app.config import settings
from app.models import AutoTradeStartRequest
from app.storage.sqlite import (
    connect_sqlite,
    is_recoverable_sqlite_error,
    sqlite_write_with_retry,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS auto_trading_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    account_key TEXT NOT NULL DEFAULT '',
    execution_mode TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL,
    max_cycles INTEGER,
    run_immediately INTEGER NOT NULL,
    auto_confirm_paper INTEGER NOT NULL,
    cycle_count INTEGER NOT NULL DEFAULT 0,
    next_run_at TEXT,
    locked_by TEXT,
    locked_at TEXT,
    locked_until TEXT,
    last_error TEXT,
    last_recoverable_error TEXT,
    last_cycle_error TEXT,
    recovery_applied INTEGER NOT NULL DEFAULT 0,
    last_results_json TEXT NOT NULL DEFAULT '[]',
    request_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auto_trading_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    results_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(session_id) REFERENCES auto_trading_sessions(session_id)
);

"""


def _connect(path: Path) -> sqlite3.Connection:
    return connect_sqlite(path)


def _write_with_retry(operation):
    return sqlite_write_with_retry(operation)


def initialize_auto_trading_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def operation() -> None:
        with _connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)

    _write_with_retry(operation)


def create_session(
    req: AutoTradeStartRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    next_run_at = now if req.run_immediately else _plus_seconds(now, req.interval_seconds)
    session_id = uuid4().hex
    request_json = _request_json(req)
    account_key = account_key_for_request(req)

    def operation() -> dict[str, Any]:
        with _connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            if settings.auto_trading_one_session_per_account:
                _expire_recoverable_error_sessions_for_account(
                    conn=conn,
                    account_key=account_key,
                    created_at=now,
                )
                existing = conn.execute(
                    """
                    SELECT *
                    FROM auto_trading_sessions
                    WHERE status = 'active'
                      AND account_key = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT 1
                    """,
                    (account_key,),
                ).fetchone()
                if existing:
                    return _row_to_session(existing)
            conn.execute(
                """
                INSERT INTO auto_trading_sessions (
                    session_id, created_at, updated_at, status, account_key, execution_mode,
                    interval_seconds, max_cycles, run_immediately, auto_confirm_paper,
                    cycle_count, next_run_at, last_results_json, request_json
                )
                VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, 0, ?, '[]', ?)
                """,
                (
                    session_id,
                    now,
                    now,
                    account_key,
                    req.execution_mode,
                    req.interval_seconds,
                    req.max_cycles,
                    1 if req.run_immediately else 0,
                    1 if req.auto_confirm_paper else 0,
                    next_run_at,
                    request_json,
                ),
            )
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="created",
                status="active",
                message="Auto-trading session created",
                results=[],
                created_at=now,
            )
            return _select_session(conn, session_id) or {}

    existing_or_created = _write_with_retry(operation)
    if existing_or_created.get("session_id") != session_id:
        return existing_or_created
    return existing_or_created


def get_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM auto_trading_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return _row_to_session(row) if row else None


def get_active_session_for_account(
    account_key: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM auto_trading_sessions
            WHERE status = 'active'
              AND account_key = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (account_key,),
        ).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(
    status: str | None = None,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    if not path.exists():
        return []
    limit = max(1, min(int(limit), 500))
    where = ""
    params: list[Any] = []
    if status:
        where = "WHERE status = ?"
        params.append(status)
    params.append(limit)
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT *
            FROM auto_trading_sessions
            {where}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_row_to_session(row) for row in rows]


def list_events(
    session_id: str,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    if not path.exists():
        return []
    limit = max(1, min(int(limit), 500))
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM auto_trading_events
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [_row_to_event(row) for row in rows]


def record_session_event(
    session_id: str,
    *,
    event_type: str,
    status: str,
    message: str,
    results: list[dict[str, Any]],
    update_last_results: bool = True,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    now = _now()
    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            row = conn.execute(
                "SELECT status FROM auto_trading_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            if update_last_results:
                conn.execute(
                    """
                    UPDATE auto_trading_sessions
                    SET updated_at = ?, last_error = NULL, last_cycle_error = NULL,
                        last_results_json = ?
                    WHERE session_id = ?
                    """,
                    (now, _json(results), session_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE auto_trading_sessions
                    SET updated_at = ?
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type=event_type,
                status=status,
                message=message,
                results=results,
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def stop_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    now = _now()
    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            row = conn.execute(
                "SELECT status FROM auto_trading_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE auto_trading_sessions
                    SET status = 'stopped', updated_at = ?, next_run_at = NULL,
                    locked_by = NULL, locked_at = NULL, locked_until = NULL
                WHERE session_id = ?
                """,
                (now, session_id),
            )
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="stopped",
                status="stopped",
                message="Auto-trading session stopped",
                results=[],
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def restart_session(
    session_id: str,
    run_immediately: bool = True,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    session = get_session(session_id, db_path=db_path)
    if not session:
        return None

    path = _db_path(db_path)
    now = _now()
    next_run_at = (
        now
        if run_immediately
        else _plus_seconds(now, int(session["interval_seconds"]))
    )
    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            conn.execute(
                """
                UPDATE auto_trading_sessions
                SET status = 'active', updated_at = ?, next_run_at = ?,
                    locked_by = NULL, locked_at = NULL, locked_until = NULL,
                    last_error = NULL, last_cycle_error = NULL,
                    recovery_applied = 1
                WHERE session_id = ?
                """,
                (now, next_run_at, session_id),
            )
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="restarted",
                status="active",
                message="Auto-trading session restarted",
                results=[],
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def claim_due_sessions(
    worker_id: str,
    limit: int = 10,
    db_path: Path | str | None = None,
    lock_seconds: int | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    now = _now()
    lock_until = _plus_seconds(now, _worker_lock_seconds(lock_seconds))

    def operation() -> list[dict[str, Any]]:
        with _connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            rows = conn.execute(
                """
                SELECT *
                FROM auto_trading_sessions
                WHERE status = 'active'
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                  AND (locked_until IS NULL OR locked_until < ?)
                ORDER BY next_run_at ASC, created_at ASC
                LIMIT ?
                """,
                (now, now, limit),
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                cursor = conn.execute(
                    """
                    UPDATE auto_trading_sessions
                    SET locked_by = ?, locked_at = ?, locked_until = ?, updated_at = ?
                    WHERE session_id = ?
                      AND status = 'active'
                      AND (locked_until IS NULL OR locked_until < ?)
                    """,
                    (worker_id, now, lock_until, now, row["session_id"], now),
                )
                if cursor.rowcount:
                    claimed.append(_row_to_session(row))
        return claimed

    return _write_with_retry(operation)


def _worker_lock_seconds(lock_seconds: int | None = None) -> int:
    configured = (
        int(lock_seconds)
        if lock_seconds is not None
        else int(settings.auto_trading_worker_lock_seconds or 0)
    )
    source_count = max(1, int(settings.universe_scanner_max_source_symbols or 1))
    interval = max(0.0, float(settings.universe_scanner_symbol_interval_seconds or 0.0))
    cap = max(0.0, float(settings.universe_scanner_symbol_interval_cap_seconds or 0.0))
    if cap > 0:
        interval = min(interval, cap)
    scanner_floor = int(source_count * interval) + 300
    return max(7200, configured, scanner_floor)


def _stale_lock_recover_seconds() -> int:
    configured = max(
        60,
        int(settings.auto_trading_stale_lock_recover_seconds or 1200),
    )
    source_count = max(1, int(settings.universe_scanner_max_source_symbols or 1))
    interval = max(0.0, float(settings.universe_scanner_symbol_interval_seconds or 0.0))
    cap = max(0.0, float(settings.universe_scanner_symbol_interval_cap_seconds or 0.0))
    if cap > 0:
        interval = min(interval, cap)
    scanner_floor = int(source_count * interval) + 600
    return max(configured, scanner_floor)


def recover_overdue_active_sessions(
    *,
    db_path: Path | str | None = None,
    min_overdue_seconds: int | float | None = None,
) -> list[dict[str, Any]]:
    """Release stale active sessions whose due schedule is stuck behind a lock."""
    path = _db_path(db_path)
    if not path.exists():
        return []
    now = _now()
    min_overdue = (
        max(60.0, float(settings.auto_trading_worker_poll_seconds or 2.0) * 5)
        if min_overdue_seconds is None
        else max(0.0, float(min_overdue_seconds))
    )
    overdue_before = _minus_seconds(now, min_overdue)
    stale_before = _minus_seconds(now, _stale_lock_recover_seconds())
    def operation() -> list[str]:
        recovered_ids: list[str] = []
        with _connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            rows = conn.execute(
                """
                SELECT session_id
                FROM auto_trading_sessions
                WHERE status = 'active'
                  AND (next_run_at IS NULL OR next_run_at <= ?)
                  AND locked_until IS NOT NULL
                  AND (
                      locked_until <= ?
                      OR (locked_at IS NOT NULL AND locked_at <= ?)
                      OR (locked_at IS NULL AND updated_at <= ?)
                  )
                """,
                (overdue_before, now, stale_before, stale_before),
            ).fetchall()
            for row in rows:
                session_id = row["session_id"]
                conn.execute(
                    """
                    UPDATE auto_trading_sessions
                    SET updated_at = ?, next_run_at = ?, locked_by = NULL,
                        locked_at = NULL, locked_until = NULL, recovery_applied = 1
                    WHERE session_id = ?
                    """,
                    (now, now, session_id),
                )
                _insert_event(
                    conn=conn,
                    session_id=session_id,
                    event_type="schedule_recovered",
                    status="active",
                    message="Recovered overdue active session schedule",
                    results=[],
                    created_at=now,
                )
                recovered_ids.append(session_id)
        return recovered_ids

    recovered_ids = _write_with_retry(operation)
    return [
        session
        for session in (get_session(session_id, db_path=path) for session_id in recovered_ids)
        if session is not None
    ]


def complete_cycle(
    session_id: str,
    results: list[dict[str, Any]],
    db_path: Path | str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any] | None:
    session = get_session(session_id, db_path=db_path)
    if not session:
        return None

    now = _now()
    cycle_count = int(session.get("cycle_count") or 0) + 1
    max_cycles = session.get("max_cycles")
    status = "stopped" if max_cycles is not None and cycle_count >= int(max_cycles) else "active"
    next_run_at = None if status == "stopped" else _plus_seconds(now, int(session["interval_seconds"]))
    path = _db_path(db_path)
    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            params: tuple[Any, ...]
            where = "WHERE session_id = ?"
            params = (
                status,
                now,
                cycle_count,
                next_run_at,
                _json(results),
                session_id,
            )
            if worker_id is not None:
                where += " AND locked_by = ?"
                params = (*params, worker_id)
            cursor = conn.execute(
                """
                UPDATE auto_trading_sessions
                SET status = ?, updated_at = ?, cycle_count = ?, next_run_at = ?,
                    locked_by = NULL, locked_at = NULL, locked_until = NULL,
                    last_error = NULL, last_cycle_error = NULL,
                    last_results_json = ?
                """ + where,
                params,
            )
            if worker_id is not None and not cursor.rowcount:
                return _select_session(conn, session_id)
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="cycle_completed",
                status=status,
                message="Auto-trading cycle completed",
                results=results,
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def recoverable_cycle_error(
    session_id: str,
    error: str,
    db_path: Path | str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any] | None:
    session = get_session(session_id, db_path=db_path)
    if not session:
        return None

    now = _now()
    cycle_count = int(session.get("cycle_count") or 0) + 1
    max_cycles = session.get("max_cycles")
    status = "stopped" if max_cycles is not None and cycle_count >= int(max_cycles) else "active"
    next_run_at = None if status == "stopped" else _plus_seconds(now, int(session["interval_seconds"]))
    path = _db_path(db_path)
    results = [
        {
            "symbol": "__cycle__",
            "status": "skipped",
            "recoverable": True,
            "message": error,
        }
    ]

    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            where = "WHERE session_id = ?"
            params: tuple[Any, ...] = (
                status,
                now,
                cycle_count,
                next_run_at,
                error,
                error,
                _json(results),
                session_id,
            )
            if worker_id is not None:
                where += " AND locked_by = ?"
                params = (*params, worker_id)
            cursor = conn.execute(
                """
                UPDATE auto_trading_sessions
                SET status = ?, updated_at = ?, cycle_count = ?, next_run_at = ?,
                    locked_by = NULL, locked_at = NULL, locked_until = NULL,
                    last_error = NULL, last_recoverable_error = ?,
                    last_cycle_error = ?, recovery_applied = 1,
                    last_results_json = ?
                """ + where,
                params,
            )
            if worker_id is not None and not cursor.rowcount:
                return _select_session(conn, session_id)
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="cycle_recoverable_error",
                status="skipped",
                message=error,
                results=results,
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def fail_cycle(
    session_id: str,
    error: str,
    db_path: Path | str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    now = _now()
    def operation() -> dict[str, Any] | None:
        with _connect(path) as conn:
            conn.executescript(SCHEMA_SQL)
            _ensure_session_columns(conn)
            where = "WHERE session_id = ?"
            params: tuple[Any, ...] = (now, error, error, session_id)
            if worker_id is not None:
                where += " AND locked_by = ?"
                params = (*params, worker_id)
            cursor = conn.execute(
                """
                UPDATE auto_trading_sessions
                SET status = 'error', updated_at = ?, next_run_at = NULL,
                    locked_by = NULL, locked_at = NULL, locked_until = NULL,
                    last_error = ?, last_cycle_error = ?
                """ + where,
                params,
            )
            if worker_id is not None and not cursor.rowcount:
                return _select_session(conn, session_id)
            _insert_event(
                conn=conn,
                session_id=session_id,
                event_type="cycle_failed",
                status="error",
                message=error,
                results=[],
                created_at=now,
            )
            return _select_session(conn, session_id)

    return _write_with_retry(operation)


def load_request(session: dict[str, Any]) -> AutoTradeStartRequest:
    payload = dict(session["request_payload"])
    if payload.get("execution_mode") == "live":
        payload["live_confirm_token"] = settings.live_trading_confirm_token
    return AutoTradeStartRequest(**payload)


def session_to_status(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "account_key": session.get("account_key"),
        "execution_mode": session["execution_mode"],
        "broker_provider": (session.get("request_payload") or {}).get(
            "broker_provider",
            "kis",
        ),
        "interval_seconds": session["interval_seconds"],
        "max_cycles": session["max_cycles"],
        "cycle_count": session["cycle_count"],
        "started_at": session["created_at"],
        "updated_at": session["updated_at"],
        "next_run_at": session["next_run_at"],
        "last_error": session["last_error"],
        "last_recoverable_error": session.get("last_recoverable_error"),
        "last_cycle_error": session.get("last_cycle_error"),
        "recovery_applied": bool(session.get("recovery_applied")),
        "last_results": session["last_results"],
        "message": "Persistent auto-trading session status",
    }


def _insert_event(
    conn: sqlite3.Connection,
    session_id: str,
    event_type: str,
    status: str,
    message: str,
    results: list[dict[str, Any]],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO auto_trading_events (
            session_id, created_at, event_type, status, message, results_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, created_at, event_type, status, message, _json(results)),
    )


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["last_results"] = _parse_json(data.get("last_results_json"), [])
    data["request_payload"] = _parse_json(data.get("request_json"), {})
    return data


def _select_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    previous_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM auto_trading_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.row_factory = previous_row_factory
    return _row_to_session(row) if row else None


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["results"] = _parse_json(data.get("results_json"), [])
    return {
        "id": data["id"],
        "session_id": data["session_id"],
        "created_at": data["created_at"],
        "event_type": data["event_type"],
        "status": data["status"],
        "message": data["message"],
        "results": data["results"],
    }


def _request_json(req: AutoTradeStartRequest) -> str:
    payload = req.model_dump()
    payload["live_confirm_token"] = None
    return _json(payload)


def account_key_for_request(req: AutoTradeStartRequest) -> str:
    account_no = (settings.kis_account_no or "local").strip() or "local"
    product_code = (settings.kis_account_product_code or "").strip()
    broker = "kis" if settings.kis_account_no else "local"
    return f"{req.execution_mode}:{broker}:{account_no}:{product_code}"


def _account_key_from_payload(payload: dict[str, Any]) -> str:
    execution_mode = str(payload.get("execution_mode") or "paper")
    account_no = (settings.kis_account_no or "local").strip() or "local"
    product_code = (settings.kis_account_product_code or "").strip()
    broker = "kis" if settings.kis_account_no else "local"
    return f"{execution_mode}:{broker}:{account_no}:{product_code}"


def _ensure_session_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "auto_trading_sessions", "account_key", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "auto_trading_sessions", "locked_at", "TEXT")
    _ensure_column(conn, "auto_trading_sessions", "last_recoverable_error", "TEXT")
    _ensure_column(conn, "auto_trading_sessions", "last_cycle_error", "TEXT")
    _ensure_column(conn, "auto_trading_sessions", "recovery_applied", "INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_auto_trading_sessions_account_status
        ON auto_trading_sessions(account_key, status, updated_at)
        """
    )
    rows = conn.execute(
        """
        SELECT session_id, request_json
        FROM auto_trading_sessions
        WHERE account_key IS NULL OR account_key = ''
        """
    ).fetchall()
    for session_id, request_json in rows:
        payload = _parse_json(request_json, {})
        conn.execute(
            """
            UPDATE auto_trading_sessions
            SET account_key = ?
            WHERE session_id = ?
            """,
            (_account_key_from_payload(payload), session_id),
        )


def _expire_recoverable_error_sessions_for_account(
    *,
    conn: sqlite3.Connection,
    account_key: str,
    created_at: str,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT session_id, last_error
        FROM auto_trading_sessions
        WHERE status = 'error'
          AND account_key = ?
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 10
        """,
        (account_key,),
    ).fetchall()

    expired: list[str] = []
    for row in rows:
        session_id = row["session_id"] if isinstance(row, sqlite3.Row) else row[0]
        last_error = row["last_error"] if isinstance(row, sqlite3.Row) else row[1]
        if not is_recoverable_sqlite_error(last_error):
            continue
        conn.execute(
            """
            UPDATE auto_trading_sessions
            SET status = 'stopped', updated_at = ?, next_run_at = NULL,
                locked_by = NULL, locked_at = NULL, locked_until = NULL,
                last_recoverable_error = COALESCE(last_recoverable_error, last_error),
                recovery_applied = 1
            WHERE session_id = ?
            """,
            (created_at, session_id),
        )
        _insert_event(
            conn=conn,
            session_id=str(session_id),
            event_type="recoverable_error_expired",
            status="stopped",
            message="Expired previous recoverable-error session before starting replacement",
            results=[],
            created_at=created_at,
        )
        expired.append(str(session_id))
    return expired


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.auto_trading_db_path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _plus_seconds(value: str, seconds: int | float) -> str:
    return (
        datetime.fromisoformat(value) + timedelta(seconds=float(seconds))
    ).isoformat(timespec="seconds")


def _minus_seconds(value: str, seconds: int | float) -> str:
    return (
        datetime.fromisoformat(value) - timedelta(seconds=float(seconds))
    ).isoformat(timespec="seconds")
