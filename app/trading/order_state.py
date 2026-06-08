from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Any

from app.config import settings
from app.models import LiveOrderRequest


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS order_intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    session_id TEXT,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    quantity_before INTEGER NOT NULL DEFAULT 0,
    quantity_after INTEGER NOT NULL DEFAULT 0,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    remaining_quantity INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    position_state_before TEXT,
    position_state_after TEXT,
    broker_order_no TEXT,
    expires_at TEXT,
    raw_request TEXT NOT NULL,
    raw_response TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_order_intents_symbol_status
ON order_intents(symbol, side, status, created_at);

CREATE TABLE IF NOT EXISTS position_states (
    symbol TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    state TEXT NOT NULL,
    current_quantity INTEGER NOT NULL DEFAULT 0,
    pending_side TEXT,
    pending_order_intent_id INTEGER,
    updated_at TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS broker_order_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    session_id TEXT,
    scan_id TEXT,
    symbol TEXT NOT NULL,
    name TEXT,
    side TEXT NOT NULL,
    qty INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    submitted_price REAL,
    notional_krw REAL,
    broker_provider TEXT NOT NULL,
    kis_is_paper INTEGER NOT NULL DEFAULT 0,
    execution_mode TEXT NOT NULL,
    broker_order_id TEXT,
    broker_response_code TEXT,
    broker_response_message TEXT,
    order_status TEXT NOT NULL,
    reject_reason TEXT,
    raw_response_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_broker_order_events_symbol_created
ON broker_order_events(symbol, created_at);

CREATE INDEX IF NOT EXISTS idx_broker_order_events_status_created
ON broker_order_events(order_status, created_at);

CREATE INDEX IF NOT EXISTS idx_broker_order_events_scan_symbol_side
ON broker_order_events(scan_id, symbol, side, order_status);
"""


PENDING_INTENT_STATUSES = ("PENDING", "SUBMITTED", "PARTIALLY_FILLED")
PENDING_POSITION_STATES = ("ENTRY_PENDING", "EXIT_PENDING", "PARTIAL")
ACTIVE_BROKER_ORDER_STATUSES = (
    "submitted",
    "accepted",
    "unknown_pending",
    "partially_filled",
)
COUNTED_BROKER_ORDER_STATUSES = (
    "submitted",
    "accepted",
    "unknown_pending",
    "partially_filled",
    "filled",
)


class OrderStateError(ValueError):
    def __init__(
        self,
        message: str,
        code: str = "order_state_rejected",
        status_code: int = 409,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def initialize_order_state_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)


def begin_order_intent(
    req: LiveOrderRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reserve a symbol transition before a live order is sent."""
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_dt()
    now_text = _format_dt(now)
    expires_at = _format_dt(
        now + timedelta(seconds=max(1, settings.order_dedupe_window_seconds))
    )
    idempotency_key = req.client_order_id or _request_fingerprint(req, now)
    target_state = "ENTRY_PENDING" if req.side == "buy" else "EXIT_PENDING"

    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        conn.execute("BEGIN IMMEDIATE")
        existing_key = _get_intent_by_key(conn, idempotency_key)
        if existing_key:
            raise OrderStateError(
                "Duplicate live order idempotency key",
                code="duplicate_idempotency_key",
            )

        position_state = _get_position_state_row(conn, req.symbol)
        state_before = position_state["state"] if position_state else "FLAT"
        current_quantity = int(position_state["current_quantity"]) if position_state else 0
        target_quantity = (
            current_quantity + int(req.quantity)
            if req.side == "buy"
            else max(0, current_quantity - int(req.quantity))
        )
        rejection = _transition_rejection(
            state=state_before,
            current_quantity=current_quantity,
            side=req.side,
        )
        if rejection:
            raise OrderStateError(rejection["message"], code=rejection["code"])

        duplicate = _find_recent_duplicate(conn, req, now)
        if duplicate:
            raise OrderStateError(
                "Recent duplicate live order is already pending or submitted",
                code="duplicate_live_order_detected",
            )

        cursor = conn.execute(
            """
            INSERT INTO order_intents (
                idempotency_key, created_at, updated_at, session_id, symbol,
                market, side, price, quantity, quantity_before, quantity_after,
                filled_quantity, remaining_quantity, status, position_state_before,
                position_state_after, expires_at, raw_request
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'PENDING', ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                now_text,
                now_text,
                req.session_id,
                req.symbol,
                req.market,
                req.side,
                req.price,
                req.quantity,
                current_quantity,
                target_quantity,
                req.quantity,
                state_before,
                target_state,
                expires_at,
                _json(req.model_dump()),
            ),
        )
        intent_id = int(cursor.lastrowid)
        _upsert_position_state(
            conn=conn,
            symbol=req.symbol,
            market=req.market,
            state=target_state,
            current_quantity=current_quantity,
            pending_side=req.side,
            pending_order_intent_id=intent_id,
            updated_at=now_text,
            raw={"source": "begin_order_intent", "state_before": state_before},
        )
        conn.commit()

    return get_order_intent(intent_id, db_path=db_path) or {}


def mark_order_submitted(
    intent_id: int,
    response: dict[str, Any],
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    broker_order_no = _extract_order_no(response)
    return _update_intent(
        intent_id=intent_id,
        status="SUBMITTED",
        response=response,
        broker_order_no=broker_order_no,
        db_path=db_path,
    )


def mark_order_failed(
    intent_id: int,
    error: str,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    now = _now()
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        conn.execute("BEGIN IMMEDIATE")
        intent = _get_intent_row(conn, intent_id)
        if not intent:
            return {}
        conn.execute(
            """
            UPDATE order_intents
            SET status = 'FAILED', updated_at = ?, last_error = ?
            WHERE id = ?
            """,
            (now, error, intent_id),
        )
        rollback_quantity = int(intent["quantity_before"] or 0)
        rollback_state = intent["position_state_before"] or _state_from_quantity(rollback_quantity)
        _upsert_position_state(
            conn=conn,
            symbol=intent["symbol"],
            market=intent["market"],
            state=rollback_state,
            current_quantity=rollback_quantity,
            pending_side=None,
            pending_order_intent_id=None,
            updated_at=now,
            raw={"source": "mark_order_failed", "error": error},
        )
        conn.commit()
    return get_order_intent(intent_id, db_path=db_path) or {}


def reconcile_after_broker_sync(
    symbol: str,
    market: str = "KR",
    account_no: str = "",
    state_db_path: Path | str | None = None,
    broker_db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile a symbol state from the latest broker sync database."""
    broker_path = settings.storage_path(broker_db_path or settings.broker_sync_db_path)
    if not broker_path.exists():
        return get_position_state(symbol, db_path=state_db_path) or {
            "symbol": symbol,
            "state": "UNKNOWN",
            "message": "No broker sync database found",
        }

    with sqlite3.connect(broker_path) as broker_conn:
        broker_conn.row_factory = sqlite3.Row
        snapshot = broker_conn.execute(
            """
            SELECT id
            FROM broker_balance_snapshots
            WHERE account_no = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (account_no,),
        ).fetchone()
        row = broker_conn.execute(
            """
            SELECT symbol, quantity, synced_at, raw_json
            FROM broker_positions
            WHERE account_no = ? AND symbol = ?
            """,
            (account_no, symbol),
        ).fetchone()

    if not snapshot and not row:
        return get_position_state(symbol, db_path=state_db_path) or {
            "symbol": symbol,
            "state": "UNKNOWN",
            "message": "No broker sync snapshot found",
        }

    quantity = int(row["quantity"]) if row else 0
    observed_state = "LONG" if quantity > 0 else "FLAT"
    now = _now()
    path = _db_path(state_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        conn.execute("BEGIN IMMEDIATE")
        current = _get_position_state_row(conn, symbol)
        current_state = current["state"] if current else observed_state
        pending_side = current["pending_side"] if current else None
        pending_intent_id = (
            int(current["pending_order_intent_id"])
            if current and current["pending_order_intent_id"] is not None
            else None
        )
        if pending_intent_id:
            intent = _get_intent_row(conn, pending_intent_id)
            pending = _pending_reconcile_result(
                intent=intent,
                observed_quantity=quantity,
                fallback_state=current_state,
            )
            if pending["pending"]:
                _upsert_position_state(
                    conn=conn,
                    symbol=symbol,
                    market=market,
                    state=pending["state"],
                    current_quantity=quantity,
                    pending_side=pending_side,
                    pending_order_intent_id=pending_intent_id,
                    updated_at=now,
                    raw={
                        "source": "broker_sync",
                        "account_no": account_no,
                        "message": pending["message"],
                    },
                )
                if intent:
                    conn.execute(
                        """
                        UPDATE order_intents
                        SET status = ?, updated_at = ?, filled_quantity = ?,
                            remaining_quantity = ?, position_state_after = ?
                        WHERE id = ?
                        """,
                        (
                            pending["intent_status"],
                            now,
                            pending["filled_quantity"],
                            pending["remaining_quantity"],
                            pending["state"],
                            pending_intent_id,
                        ),
                    )
                conn.commit()
                return get_position_state(symbol, db_path=state_db_path) or {}

        _upsert_position_state(
            conn=conn,
            symbol=symbol,
            market=market,
            state=observed_state,
            current_quantity=quantity,
            pending_side=None,
            pending_order_intent_id=None,
            updated_at=now,
            raw={
                "source": "broker_sync",
                "account_no": account_no,
                "synced_at": row["synced_at"] if row else None,
            },
        )
        if pending_intent_id:
            intent = _get_intent_row(conn, pending_intent_id)
            filled_quantity = _filled_quantity_from_observed(intent, quantity)
            conn.execute(
                """
                UPDATE order_intents
                SET status = 'FILLED', updated_at = ?, filled_quantity = ?,
                    remaining_quantity = 0, position_state_after = ?
                WHERE id = ? AND status IN ('PENDING', 'SUBMITTED', 'PARTIALLY_FILLED')
                """,
                (now, filled_quantity, observed_state, pending_intent_id),
            )
        conn.commit()

    return get_position_state(symbol, db_path=state_db_path) or {}


def reconcile_all_after_broker_sync(
    account_no: str = "",
    state_db_path: Path | str | None = None,
    broker_db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile every known order-state symbol from the latest broker snapshot."""
    broker_path = settings.storage_path(broker_db_path or settings.broker_sync_db_path)
    state_path = _db_path(state_db_path)
    symbols: set[str] = set()

    if broker_path.exists():
        with sqlite3.connect(broker_path) as broker_conn:
            broker_conn.row_factory = sqlite3.Row
            broker_conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS broker_positions (
                    broker TEXT NOT NULL,
                    account_no TEXT NOT NULL DEFAULT '',
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL DEFAULT 0,
                    synced_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY (broker, account_no, symbol)
                );
                """
            )
            rows = broker_conn.execute(
                "SELECT symbol FROM broker_positions WHERE account_no = ?",
                (account_no,),
            ).fetchall()
            symbols.update(str(row["symbol"]) for row in rows if row["symbol"])

    if state_path.exists():
        with sqlite3.connect(state_path) as state_conn:
            state_conn.row_factory = sqlite3.Row
            state_conn.executescript(SCHEMA_SQL)
            _ensure_order_state_columns(state_conn)
            rows = state_conn.execute(
                """
                SELECT symbol
                FROM position_states
                UNION
                SELECT symbol
                FROM order_intents
                WHERE status IN ('PENDING', 'SUBMITTED', 'PARTIALLY_FILLED')
                """
            ).fetchall()
            symbols.update(str(row["symbol"]) for row in rows if row["symbol"])

    reconciled = [
        reconcile_after_broker_sync(
            symbol=symbol,
            account_no=account_no,
            state_db_path=state_db_path,
            broker_db_path=broker_db_path,
        )
        for symbol in sorted(symbols)
    ]
    return {
        "status": "success",
        "account_no": account_no,
        "reconciled_count": len(reconciled),
        "positions": reconciled,
    }


def record_broker_order_event(
    event: dict[str, Any],
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Persist an actual broker submission event for status/dashboard counts."""
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    created_at = str(event.get("created_at") or now)
    updated_at = str(event.get("updated_at") or now)
    raw_response = event.get("raw_response_json")
    if not isinstance(raw_response, str):
        raw_response = _json(event.get("raw_response") or {})

    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        cursor = conn.execute(
            """
            INSERT INTO broker_order_events (
                created_at, updated_at, session_id, scan_id, symbol, name,
                side, qty, order_type, limit_price, submitted_price, notional_krw,
                broker_provider, kis_is_paper, execution_mode, broker_order_id,
                broker_response_code,
                broker_response_message, order_status, reject_reason,
                raw_response_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                updated_at,
                event.get("session_id"),
                event.get("scan_id"),
                str(event.get("symbol") or ""),
                event.get("name"),
                str(event.get("side") or "").lower(),
                int(event.get("qty") or event.get("quantity") or 0),
                str(event.get("order_type") or "limit"),
                event.get("limit_price"),
                event.get("submitted_price"),
                event.get("notional_krw")
                if event.get("notional_krw") is not None
                else _event_notional(event),
                str(event.get("broker_provider") or "kis").lower(),
                1 if bool(event.get("kis_is_paper")) else 0,
                str(event.get("execution_mode") or ""),
                event.get("broker_order_id") or event.get("broker_order_no"),
                event.get("broker_response_code"),
                event.get("broker_response_message"),
                str(event.get("order_status") or "unknown_pending").lower(),
                event.get("reject_reason"),
                raw_response,
            ),
        )
        conn.commit()
        event_id = int(cursor.lastrowid)
    return get_broker_order_event(event_id, db_path=db_path) or {}


def get_broker_order_event(
    event_id: int,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        row = conn.execute(
            "SELECT * FROM broker_order_events WHERE id = ?",
            (event_id,),
        ).fetchone()
    return _broker_order_event_to_dict(row) if row else None


def latest_broker_order_event(
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        row = conn.execute(
            """
            SELECT *
            FROM broker_order_events
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    return _broker_order_event_to_dict(row) if row else None


def get_order_intent(
    intent_id: int,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        row = _get_intent_row(conn, intent_id)
    return _intent_to_dict(row) if row else None


def get_position_state(
    symbol: str,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    path = _db_path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        row = _get_position_state_row(conn, symbol)
    return _position_state_to_dict(row) if row else None


def broker_paper_order_risk_limits() -> dict[str, Any]:
    return settings.broker_paper_risk_limits()


def validate_broker_paper_order(
    req: LiveOrderRequest,
    *,
    name: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return whether a broker_paper order passes DB-backed duplicate/risk limits."""
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_dt()
    today_start = datetime(now.year, now.month, now.day)
    limits = broker_paper_order_risk_limits()
    notional = float(req.price) * int(req.quantity)

    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        conn.execute("BEGIN IMMEDIATE")
        block = _broker_paper_order_block(
            conn=conn,
            req=req,
            notional=notional,
            now=now,
            today_start=today_start,
            limits=limits,
        )
        conn.commit()

    status = {
        "approved": block is None,
        "broker_submit_blocked": block is not None,
        "broker_submit_block_reason": None if block is None else block["reason"],
        "broker_submit_block_code": None if block is None else block["code"],
        "notional_krw": notional,
        "broker_order_risk_limits": limits,
        "symbol": req.symbol,
        "side": req.side,
        "scan_id": req.scan_id,
        "session_id": req.session_id,
        "name": name,
    }
    if block:
        status.update(block.get("details") or {})
    return status


def broker_paper_order_risk_snapshot(
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    limits = broker_paper_order_risk_limits()
    default = {
        "broker_order_risk_limits": limits,
        "last_order_by_symbol": {},
        "open_broker_order_count": 0,
        "today_broker_order_count": 0,
        "today_broker_notional_krw": 0.0,
    }
    if not path.exists():
        return default
    try:
        now = _now_dt()
        today_start = _format_dt(datetime(now.year, now.month, now.day))
        active_statuses = ",".join("?" for _ in ACTIVE_BROKER_ORDER_STATUSES)
        counted_statuses = ",".join("?" for _ in COUNTED_BROKER_ORDER_STATUSES)
        with sqlite3.connect(path, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            if not _table_exists(conn, "broker_order_events"):
                return default
            _ensure_order_state_columns(conn)
            open_count = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM broker_order_events
                WHERE execution_mode = 'broker_paper'
                  AND order_status IN ({active_statuses})
                """,
                ACTIVE_BROKER_ORDER_STATUSES,
            ).fetchone()[0]
            today = conn.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(notional_krw), 0)
                FROM broker_order_events
                WHERE execution_mode = 'broker_paper'
                  AND order_status IN ({counted_statuses})
                  AND created_at >= ?
                """,
                (*COUNTED_BROKER_ORDER_STATUSES, today_start),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT symbol, created_at, side, qty, submitted_price,
                       notional_krw, broker_order_id, order_status, scan_id
                FROM broker_order_events
                WHERE execution_mode = 'broker_paper'
                  AND symbol IS NOT NULL
                  AND symbol <> ''
                ORDER BY created_at DESC, id DESC
                LIMIT 200
                """
            ).fetchall()
    except sqlite3.Error:
        return default
    last_by_symbol: dict[str, Any] = {}
    for row in rows:
        symbol = str(row["symbol"])
        if symbol in last_by_symbol:
            continue
        last_by_symbol[symbol] = dict(row)
    return {
        "broker_order_risk_limits": limits,
        "last_order_by_symbol": last_by_symbol,
        "open_broker_order_count": int(open_count or 0),
        "today_broker_order_count": int(today[0] or 0) if today else 0,
        "today_broker_notional_krw": float(today[1] or 0.0) if today else 0.0,
    }


def _update_intent(
    intent_id: int,
    status: str,
    response: dict[str, Any] | None = None,
    broker_order_no: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    now = _now()
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        _ensure_order_state_columns(conn)
        conn.execute(
            """
            UPDATE order_intents
            SET status = ?, updated_at = ?, raw_response = ?,
                broker_order_no = COALESCE(?, broker_order_no)
            WHERE id = ?
            """,
            (
                status,
                now,
                _json(response) if response is not None else None,
                broker_order_no,
                intent_id,
            ),
        )
        conn.commit()
    return get_order_intent(intent_id, db_path=db_path) or {}


def _pending_reconcile_result(
    intent: sqlite3.Row | None,
    observed_quantity: int,
    fallback_state: str,
) -> dict[str, Any]:
    if not intent:
        return {
            "pending": False,
            "state": _state_from_quantity(observed_quantity),
            "intent_status": "UNKNOWN",
            "filled_quantity": 0,
            "remaining_quantity": 0,
            "message": "No pending intent row found",
        }

    before = int(intent["quantity_before"] or 0)
    requested = int(intent["quantity"] or 0)
    side = str(intent["side"])
    filled = _filled_quantity_from_observed(intent, observed_quantity)
    remaining = max(0, requested - filled)

    if filled <= 0:
        return {
            "pending": True,
            "state": fallback_state
            if fallback_state in PENDING_POSITION_STATES
            else ("ENTRY_PENDING" if side == "buy" else "EXIT_PENDING"),
            "intent_status": "SUBMITTED",
            "filled_quantity": 0,
            "remaining_quantity": requested,
            "message": "Fill not confirmed yet",
        }

    if remaining > 0:
        return {
            "pending": True,
            "state": "PARTIAL",
            "intent_status": "PARTIALLY_FILLED",
            "filled_quantity": filled,
            "remaining_quantity": remaining,
            "message": "Order is partially filled",
        }

    return {
        "pending": False,
        "state": _state_from_quantity(observed_quantity),
        "intent_status": "FILLED",
        "filled_quantity": requested,
        "remaining_quantity": 0,
        "message": "Order fill confirmed",
    }


def _filled_quantity_from_observed(
    intent: sqlite3.Row | None,
    observed_quantity: int,
) -> int:
    if not intent:
        return 0
    before = int(intent["quantity_before"] or 0)
    requested = int(intent["quantity"] or 0)
    if str(intent["side"]) == "buy":
        return max(0, min(requested, observed_quantity - before))
    return max(0, min(requested, before - observed_quantity))


def _state_from_quantity(quantity: int) -> str:
    return "LONG" if int(quantity or 0) > 0 else "FLAT"


def _extract_order_no(response: dict[str, Any]) -> str | None:
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, dict):
        for key in ("ODNO", "odno", "order_no"):
            if output.get(key):
                return str(output[key])
    for key in ("ODNO", "odno", "order_no"):
        if response.get(key):
            return str(response[key])
    return None


def _transition_rejection(
    state: str,
    current_quantity: int,
    side: str,
) -> dict[str, str] | None:
    if state in PENDING_POSITION_STATES:
        return {
            "code": "position_transition_pending",
            "message": f"Position state is already {state}",
        }
    if side == "buy" and state == "LONG" and not settings.allow_position_additions:
        return {
            "code": "position_already_long",
            "message": "Position is already LONG; additions are disabled",
        }
    if side == "sell" and state == "FLAT" and current_quantity <= 0:
        return {
            "code": "position_already_flat",
            "message": "Position is already FLAT; sell order is blocked",
        }
    return None


def _broker_paper_order_block(
    *,
    conn: sqlite3.Connection,
    req: LiveOrderRequest,
    notional: float,
    now: datetime,
    today_start: datetime,
    limits: dict[str, Any],
) -> dict[str, Any] | None:
    if notional <= 0:
        return _broker_block("notional_zero", "Order notional is zero")

    max_order = float(limits.get("max_order_krw") or 0.0)
    if max_order and notional > max_order:
        return _broker_block(
            "max_order_notional_exceeded",
            "Broker paper max order notional exceeded",
            {"notional_krw": notional, "max_order_krw": max_order},
        )

    if req.side == "buy":
        active_statuses = ",".join("?" for _ in ACTIVE_BROKER_ORDER_STATUSES)
        open_event = conn.execute(
            f"""
            SELECT *
            FROM broker_order_events
            WHERE execution_mode = 'broker_paper'
              AND symbol = ?
              AND side = 'buy'
              AND order_status IN ({active_statuses})
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (req.symbol, *ACTIVE_BROKER_ORDER_STATUSES),
        ).fetchone()
        if open_event:
            return _broker_block(
                "open_broker_order_exists",
                "Open broker order already exists for this symbol",
                {"existing_broker_order_id": open_event["broker_order_id"]},
            )

        intent_statuses = ",".join("?" for _ in PENDING_INTENT_STATUSES)
        open_intent = conn.execute(
            f"""
            SELECT *
            FROM order_intents
            WHERE symbol = ?
              AND side = 'buy'
              AND status IN ({intent_statuses})
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (req.symbol, *PENDING_INTENT_STATUSES),
        ).fetchone()
        if open_intent:
            return _broker_block(
                "open_order_intent_exists",
                "Open order intent already exists for this symbol",
                {"existing_intent_id": open_intent["id"]},
            )

        position_state = _get_position_state_row(conn, req.symbol)
        if (
            position_state
            and str(position_state["state"]) == "LONG"
            and not settings.allow_position_additions
        ):
            return _broker_block(
                "already_position_exists",
                "Position is already LONG; additions are disabled",
                {"current_quantity": int(position_state["current_quantity"] or 0)},
            )

    if req.scan_id:
        duplicate_scan = conn.execute(
            """
            SELECT *
            FROM broker_order_events
            WHERE scan_id = ?
              AND symbol = ?
              AND side = ?
              AND order_status IN (
                'submitted', 'accepted', 'unknown_pending',
                'partially_filled', 'filled'
              )
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (req.scan_id, req.symbol, req.side),
        ).fetchone()
        if duplicate_scan:
            return _broker_block(
                "duplicate_scan_symbol_side",
                "Broker order already exists for this scan, symbol, and side",
                {"scan_id": req.scan_id},
            )

    cooldown_days = int(limits.get("symbol_cooldown_days") or 0)
    if cooldown_days > 0 and req.side == "buy":
        since = _format_dt(now - timedelta(days=cooldown_days))
        recent_symbol = conn.execute(
            """
            SELECT *
            FROM broker_order_events
            WHERE execution_mode = 'broker_paper'
              AND symbol = ?
              AND side = 'buy'
              AND order_status IN (
                'submitted', 'accepted', 'unknown_pending',
                'partially_filled', 'filled'
              )
              AND created_at >= ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (req.symbol, since),
        ).fetchone()
        if recent_symbol:
            return _broker_block(
                "symbol_cooldown_active",
                "Broker paper symbol cooldown is active",
                {
                    "cooldown_days": cooldown_days,
                    "last_order_at": recent_symbol["created_at"],
                },
            )

    today_text = _format_dt(today_start)
    counted_statuses = ",".join("?" for _ in COUNTED_BROKER_ORDER_STATUSES)
    daily = conn.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(notional_krw), 0)
        FROM broker_order_events
        WHERE execution_mode = 'broker_paper'
          AND order_status IN ({counted_statuses})
          AND created_at >= ?
        """,
        (*COUNTED_BROKER_ORDER_STATUSES, today_text),
    ).fetchone()
    daily_count = int(daily[0] or 0) if daily else 0
    daily_notional = float(daily[1] or 0.0) if daily else 0.0
    max_daily_orders = int(limits.get("max_daily_orders") or 0)
    if max_daily_orders and daily_count >= max_daily_orders:
        return _broker_block(
            "daily_order_limit_exceeded",
            "Broker paper daily order limit reached",
            {"today_broker_order_count": daily_count, "max_daily_orders": max_daily_orders},
        )

    max_daily_notional = float(limits.get("max_daily_notional_krw") or 0.0)
    if max_daily_notional and daily_notional + notional > max_daily_notional:
        return _broker_block(
            "daily_notional_limit_exceeded",
            "Broker paper daily notional limit exceeded",
            {
                "today_broker_notional_krw": daily_notional,
                "order_notional_krw": notional,
                "max_daily_notional_krw": max_daily_notional,
            },
        )

    per_symbol = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM broker_order_events
        WHERE execution_mode = 'broker_paper'
          AND symbol = ?
          AND order_status IN ({counted_statuses})
          AND created_at >= ?
        """,
        (req.symbol, *COUNTED_BROKER_ORDER_STATUSES, today_text),
    ).fetchone()
    per_symbol_count = int(per_symbol[0] or 0) if per_symbol else 0
    max_per_symbol = int(limits.get("max_daily_orders_per_symbol") or 0)
    if max_per_symbol and per_symbol_count >= max_per_symbol:
        return _broker_block(
            "daily_symbol_order_limit_exceeded",
            "Broker paper daily per-symbol order limit reached",
            {
                "today_symbol_order_count": per_symbol_count,
                "max_daily_orders_per_symbol": max_per_symbol,
            },
        )

    return None


def _broker_block(
    code: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"code": code, "reason": reason, "details": details or {}}


def _event_notional(event: dict[str, Any]) -> float | None:
    price = event.get("submitted_price") or event.get("limit_price")
    qty = event.get("qty") or event.get("quantity")
    try:
        return float(price) * int(qty)
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return bool(row)


def _ensure_order_state_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "order_intents", "quantity_before", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "order_intents", "quantity_after", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "order_intents", "filled_quantity", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "order_intents", "remaining_quantity", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "order_intents", "broker_order_no", "TEXT")
    _ensure_column(conn, "broker_order_events", "session_id", "TEXT")
    _ensure_column(conn, "broker_order_events", "scan_id", "TEXT")
    _ensure_column(conn, "broker_order_events", "name", "TEXT")
    _ensure_column(conn, "broker_order_events", "notional_krw", "REAL")


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _find_recent_duplicate(
    conn: sqlite3.Connection,
    req: LiveOrderRequest,
    now: datetime,
) -> sqlite3.Row | None:
    since = _format_dt(
        now - timedelta(seconds=max(1, settings.order_dedupe_window_seconds))
    )
    statuses = ",".join("?" for _ in PENDING_INTENT_STATUSES)
    return conn.execute(
        f"""
        SELECT *
        FROM order_intents
        WHERE symbol = ?
          AND side = ?
          AND status IN ({statuses})
          AND created_at >= ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (req.symbol, req.side, *PENDING_INTENT_STATUSES, since),
    ).fetchone()


def _get_intent_by_key(
    conn: sqlite3.Connection,
    idempotency_key: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM order_intents WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()


def _get_intent_row(conn: sqlite3.Connection, intent_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM order_intents WHERE id = ?",
        (intent_id,),
    ).fetchone()


def _get_position_state_row(
    conn: sqlite3.Connection,
    symbol: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM position_states WHERE symbol = ?",
        (symbol,),
    ).fetchone()


def _upsert_position_state(
    conn: sqlite3.Connection,
    symbol: str,
    market: str,
    state: str,
    current_quantity: int,
    pending_side: str | None,
    pending_order_intent_id: int | None,
    updated_at: str,
    raw: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO position_states (
            symbol, market, state, current_quantity, pending_side,
            pending_order_intent_id, updated_at, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            market = excluded.market,
            state = excluded.state,
            current_quantity = excluded.current_quantity,
            pending_side = excluded.pending_side,
            pending_order_intent_id = excluded.pending_order_intent_id,
            updated_at = excluded.updated_at,
            raw_json = excluded.raw_json
        """,
        (
            symbol,
            market,
            state,
            current_quantity,
            pending_side,
            pending_order_intent_id,
            updated_at,
            _json(raw),
        ),
    )


def _intent_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["raw_request"] = _parse_json(data.get("raw_request"), {})
    data["raw_response"] = _parse_json(data.get("raw_response"), None)
    return data


def _broker_order_event_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["kis_is_paper"] = bool(data.get("kis_is_paper"))
    data["raw_response"] = _parse_json(data.pop("raw_response_json", None), {})
    return data


def _position_state_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["raw"] = _parse_json(data.pop("raw_json", None), {})
    return data


def _request_fingerprint(req: LiveOrderRequest, now: datetime) -> str:
    bucket_seconds = max(1, settings.order_dedupe_window_seconds)
    bucket = int(now.timestamp() // bucket_seconds)
    payload = {
        "bucket": bucket,
        "session_id": req.session_id,
        "symbol": req.symbol,
        "side": req.side,
        "price": req.price,
        "quantity": req.quantity,
        "scan_id": req.scan_id,
    }
    digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
    return f"live:{digest[:32]}"


def _quantity_for_state(state: str) -> int:
    return 1 if state == "LONG" else 0


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.order_state_db_path)


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _now_dt() -> datetime:
    return datetime.now()


def _now() -> str:
    return _format_dt(_now_dt())


def _format_dt(value: datetime) -> str:
    return value.isoformat(timespec="seconds")
