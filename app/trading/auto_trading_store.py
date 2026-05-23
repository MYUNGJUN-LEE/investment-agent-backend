from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any
from uuid import uuid4

from app.config import settings
from app.models import AutoTradeStartRequest


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS auto_trading_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL,
    max_cycles INTEGER,
    run_immediately INTEGER NOT NULL,
    auto_confirm_paper INTEGER NOT NULL,
    cycle_count INTEGER NOT NULL DEFAULT 0,
    next_run_at TEXT,
    locked_by TEXT,
    locked_until TEXT,
    last_error TEXT,
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


def initialize_auto_trading_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


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

    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            """
            INSERT INTO auto_trading_sessions (
                session_id, created_at, updated_at, status, execution_mode,
                interval_seconds, max_cycles, run_immediately, auto_confirm_paper,
                cycle_count, next_run_at, last_results_json, request_json
            )
            VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, 0, ?, '[]', ?)
            """,
            (
                session_id,
                now,
                now,
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
    return get_session(session_id, db_path=db_path) or {}


def get_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        row = conn.execute(
            "SELECT * FROM auto_trading_sessions WHERE session_id = ?",
            (session_id,),
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
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
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
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
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


def stop_session(
    session_id: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
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
                locked_by = NULL, locked_until = NULL
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
    return get_session(session_id, db_path=db_path)


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
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            """
            UPDATE auto_trading_sessions
            SET status = 'active', updated_at = ?, next_run_at = ?,
                locked_by = NULL, locked_until = NULL, last_error = NULL
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
    return get_session(session_id, db_path=db_path)


def claim_due_sessions(
    worker_id: str,
    limit: int = 10,
    db_path: Path | str | None = None,
    lock_seconds: int | None = None,
) -> list[dict[str, Any]]:
    path = _db_path(db_path)
    now = _now()
    lock_until = _plus_seconds(now, lock_seconds or settings.auto_trading_worker_lock_seconds)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
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
                SET locked_by = ?, locked_until = ?, updated_at = ?
                WHERE session_id = ?
                  AND status = 'active'
                  AND (locked_until IS NULL OR locked_until < ?)
                """,
                (worker_id, lock_until, now, row["session_id"], now),
            )
            if cursor.rowcount:
                claimed.append(_row_to_session(row))
    return claimed


def complete_cycle(
    session_id: str,
    results: list[dict[str, Any]],
    db_path: Path | str | None = None,
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
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            """
            UPDATE auto_trading_sessions
            SET status = ?, updated_at = ?, cycle_count = ?, next_run_at = ?,
                locked_by = NULL, locked_until = NULL, last_error = NULL,
                last_results_json = ?
            WHERE session_id = ?
            """,
            (
                status,
                now,
                cycle_count,
                next_run_at,
                _json(results),
                session_id,
            ),
        )
        _insert_event(
            conn=conn,
            session_id=session_id,
            event_type="cycle_completed",
            status=status,
            message="Auto-trading cycle completed",
            results=results,
            created_at=now,
        )
    return get_session(session_id, db_path=db_path)


def fail_cycle(
    session_id: str,
    error: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    now = _now()
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            """
            UPDATE auto_trading_sessions
            SET status = 'error', updated_at = ?, next_run_at = NULL,
                locked_by = NULL, locked_until = NULL, last_error = ?
            WHERE session_id = ?
            """,
            (now, error, session_id),
        )
        _insert_event(
            conn=conn,
            session_id=session_id,
            event_type="cycle_failed",
            status="error",
            message=error,
            results=[],
            created_at=now,
        )
    return get_session(session_id, db_path=db_path)


def load_request(session: dict[str, Any]) -> AutoTradeStartRequest:
    payload = dict(session["request_payload"])
    if payload.get("execution_mode") == "live":
        payload["live_confirm_token"] = settings.live_trading_confirm_token
    return AutoTradeStartRequest(**payload)


def session_to_status(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "execution_mode": session["execution_mode"],
        "interval_seconds": session["interval_seconds"],
        "max_cycles": session["max_cycles"],
        "cycle_count": session["cycle_count"],
        "started_at": session["created_at"],
        "updated_at": session["updated_at"],
        "next_run_at": session["next_run_at"],
        "last_error": session["last_error"],
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
    return Path(db_path or settings.auto_trading_db_path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _plus_seconds(value: str, seconds: int | float) -> str:
    return (
        datetime.fromisoformat(value) + timedelta(seconds=float(seconds))
    ).isoformat(timespec="seconds")
