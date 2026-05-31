from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fill_quality_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    decision_price REAL,
    order_price REAL,
    filled_price REAL,
    requested_quantity REAL,
    filled_quantity REAL,
    fill_ratio REAL,
    slippage_bps REAL,
    fill_delay_seconds REAL,
    execution_mode TEXT,
    source TEXT,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fill_quality_events_symbol_time
ON fill_quality_events(symbol, created_at);

CREATE TABLE IF NOT EXISTS fill_quality_stats (
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    ewma_slippage_bps REAL NOT NULL,
    avg_slippage_bps REAL NOT NULL,
    ewma_fill_ratio REAL NOT NULL,
    avg_fill_ratio REAL NOT NULL,
    ewma_fill_delay_seconds REAL NOT NULL,
    avg_fill_delay_seconds REAL NOT NULL,
    updated_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY(symbol, side)
);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.fill_quality_db_path)


def initialize_fill_quality_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
    finally:
        conn.close()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip()


def _normalize_side(value: Any) -> str:
    side = str(value or "buy").strip().lower()
    if side in {"sell", "exit"}:
        return "sell"
    return "buy"


def _slippage_bps(
    *,
    side: str,
    reference_price: float | None,
    filled_price: float | None,
) -> float | None:
    if reference_price is None or filled_price is None:
        return None
    if reference_price <= 0 or filled_price <= 0:
        return None

    if side == "sell":
        return (reference_price - filled_price) / reference_price * 10_000.0

    return (filled_price - reference_price) / reference_price * 10_000.0


def _fill_ratio(
    *,
    requested_quantity: float | None,
    filled_quantity: float | None,
) -> float | None:
    if requested_quantity is None or requested_quantity <= 0:
        return None
    if filled_quantity is None:
        return None
    return max(0.0, min(1.0, filled_quantity / requested_quantity))


def _ewma(previous: float, value: float, alpha: float) -> float:
    alpha = max(0.01, min(1.0, float(alpha)))
    return previous * (1.0 - alpha) + value * alpha


def _first_value(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)

    for nested_key in ("paper_result", "execution", "order_state", "live_result"):
        nested = data.get(nested_key)
        if not isinstance(nested, dict):
            continue
        value = _first_value(nested, keys)
        if value is not None:
            return value

    return None


def fill_quality_event_from_order(order: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _normalize_symbol(_first_value(order, ("symbol",)))
    if not symbol:
        return None

    side = _normalize_side(
        _first_value(order, ("side", "requested_action", "action"))
    )

    decision_price = _to_float(_first_value(order, ("decision_price",)))
    order_price = _to_float(_first_value(order, ("order_price", "price")))
    filled_price = (
        _to_float(_first_value(order, ("filled_price", "fill_price")))
        or _to_float(_first_value(order, ("avg_fill_price", "average_price")))
        or _to_float(_first_value(order, ("execution_price", "effective_price")))
    )

    requested_quantity = (
        _to_float(_first_value(order, ("requested_quantity",)))
        or _to_float(_first_value(order, ("order_quantity", "quantity")))
    )
    filled_quantity = (
        _to_float(_first_value(order, ("filled_quantity",)))
        or _to_float(_first_value(order, ("executed_quantity", "fill_quantity")))
        or _to_float(_first_value(order, ("filled_qty",)))
    )

    reference_price = decision_price or order_price
    slippage = _to_float(_first_value(order, ("slippage_bps",)))
    if slippage is None:
        slippage = _slippage_bps(
            side=side,
            reference_price=reference_price,
            filled_price=filled_price,
        )
    ratio = _fill_ratio(
        requested_quantity=requested_quantity,
        filled_quantity=filled_quantity,
    )

    fill_delay_seconds = _to_float(_first_value(order, ("fill_delay_seconds",)))

    if slippage is None and ratio is None:
        return None

    return {
        "created_at": _now(),
        "symbol": symbol,
        "side": side,
        "decision_price": decision_price,
        "order_price": order_price,
        "filled_price": filled_price,
        "requested_quantity": requested_quantity,
        "filled_quantity": filled_quantity,
        "fill_ratio": ratio,
        "slippage_bps": slippage,
        "fill_delay_seconds": fill_delay_seconds,
        "execution_mode": str(order.get("execution_mode") or ""),
        "source": str(order.get("source") or "order_result"),
        "raw_json": order,
    }


def record_fill_quality_event(
    order: dict[str, Any],
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.fill_quality_feedback_enabled):
        return {"status": "disabled"}

    event = fill_quality_event_from_order(order)
    if event is None:
        return {"status": "skipped", "message": "No fill-quality data available"}

    initialize_fill_quality_db(db_path)
    path = _db_path(db_path)

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO fill_quality_events (
                created_at, symbol, side, decision_price, order_price,
                filled_price, requested_quantity, filled_quantity,
                fill_ratio, slippage_bps, fill_delay_seconds,
                execution_mode, source, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["created_at"],
                event["symbol"],
                event["side"],
                event["decision_price"],
                event["order_price"],
                event["filled_price"],
                event["requested_quantity"],
                event["filled_quantity"],
                event["fill_ratio"],
                event["slippage_bps"],
                event["fill_delay_seconds"],
                event["execution_mode"],
                event["source"],
                json.dumps(event["raw_json"], ensure_ascii=False, default=str),
            ),
        )
        _update_fill_quality_stats(conn, event)
        _prune_recent_events(conn, symbol=event["symbol"], side=event["side"])
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "recorded",
        "symbol": event["symbol"],
        "side": event["side"],
        "slippage_bps": event["slippage_bps"],
        "fill_ratio": event["fill_ratio"],
    }


def _update_fill_quality_stats(
    conn: sqlite3.Connection,
    event: dict[str, Any],
) -> None:
    symbol = event["symbol"]
    side = event["side"]

    row = conn.execute(
        """
        SELECT *
        FROM fill_quality_stats
        WHERE symbol = ? AND side = ?
        """,
        (symbol, side),
    ).fetchone()

    alpha = float(settings.fill_quality_ewma_alpha or 0.15)

    slippage = float(event["slippage_bps"] or 0.0)
    fill_ratio = (
        float(event["fill_ratio"])
        if event["fill_ratio"] is not None
        else 1.0
    )
    delay = float(event["fill_delay_seconds"] or 0.0)

    updated_at = _now()

    if row is None:
        payload = {
            "last_event": event,
        }
        conn.execute(
            """
            INSERT INTO fill_quality_stats (
                symbol, side, sample_count, ewma_slippage_bps,
                avg_slippage_bps, ewma_fill_ratio, avg_fill_ratio,
                ewma_fill_delay_seconds, avg_fill_delay_seconds,
                updated_at, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                side,
                1,
                slippage,
                slippage,
                fill_ratio,
                fill_ratio,
                delay,
                delay,
                updated_at,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        return

    sample_count = int(row[2]) + 1
    prev_ewma_slippage = float(row[3] or 0.0)
    prev_avg_slippage = float(row[4] or 0.0)
    prev_ewma_fill_ratio = float(row[5] or 1.0)
    prev_avg_fill_ratio = float(row[6] or 1.0)
    prev_ewma_delay = float(row[7] or 0.0)
    prev_avg_delay = float(row[8] or 0.0)

    avg_slippage = (
        prev_avg_slippage * (sample_count - 1) + slippage
    ) / sample_count
    avg_fill_ratio = (
        prev_avg_fill_ratio * (sample_count - 1) + fill_ratio
    ) / sample_count
    avg_delay = (
        prev_avg_delay * (sample_count - 1) + delay
    ) / sample_count

    payload = {
        "last_event": event,
    }

    conn.execute(
        """
        UPDATE fill_quality_stats
        SET sample_count = ?,
            ewma_slippage_bps = ?,
            avg_slippage_bps = ?,
            ewma_fill_ratio = ?,
            avg_fill_ratio = ?,
            ewma_fill_delay_seconds = ?,
            avg_fill_delay_seconds = ?,
            updated_at = ?,
            raw_json = ?
        WHERE symbol = ? AND side = ?
        """,
        (
            sample_count,
            _ewma(prev_ewma_slippage, slippage, alpha),
            avg_slippage,
            _ewma(prev_ewma_fill_ratio, fill_ratio, alpha),
            avg_fill_ratio,
            _ewma(prev_ewma_delay, delay, alpha),
            avg_delay,
            updated_at,
            json.dumps(payload, ensure_ascii=False, default=str),
            symbol,
            side,
        ),
    )


def _prune_recent_events(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
) -> None:
    limit = max(10, int(settings.fill_quality_max_recent_orders or 200))
    conn.execute(
        """
        DELETE FROM fill_quality_events
        WHERE symbol = ?
          AND side = ?
          AND id NOT IN (
              SELECT id
              FROM fill_quality_events
              WHERE symbol = ?
                AND side = ?
              ORDER BY created_at DESC, id DESC
              LIMIT ?
          )
        """,
        (symbol, side, symbol, side, limit),
    )


def get_fill_quality_stats(
    symbol: str,
    *,
    side: str = "buy",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.fill_quality_feedback_enabled):
        return {"status": "disabled"}

    symbol = _normalize_symbol(symbol)
    side = _normalize_side(side)

    if not symbol:
        return {"status": "missing_symbol"}

    path = _db_path(db_path)
    if not path.exists():
        return {"status": "empty"}

    try:
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM fill_quality_stats
                WHERE symbol = ? AND side = ?
                """,
                (symbol, side),
            ).fetchone()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return {"status": "error"}

    if not row:
        return {"status": "empty", "symbol": symbol, "side": side}

    return {
        "status": "ready",
        "symbol": row["symbol"],
        "side": row["side"],
        "sample_count": int(row["sample_count"] or 0),
        "ewma_slippage_bps": float(row["ewma_slippage_bps"] or 0.0),
        "avg_slippage_bps": float(row["avg_slippage_bps"] or 0.0),
        "ewma_fill_ratio": float(row["ewma_fill_ratio"] or 1.0),
        "avg_fill_ratio": float(row["avg_fill_ratio"] or 1.0),
        "ewma_fill_delay_seconds": float(row["ewma_fill_delay_seconds"] or 0.0),
        "avg_fill_delay_seconds": float(row["avg_fill_delay_seconds"] or 0.0),
        "updated_at": row["updated_at"],
    }


def fill_quality_adjustment_for_candidate(
    candidate: dict[str, Any],
    *,
    execution_mode: str = "paper",
) -> dict[str, Any]:
    symbol = _normalize_symbol(candidate.get("symbol"))
    stats = get_fill_quality_stats(symbol, side="buy")

    if stats.get("status") != "ready":
        return {
            "status": stats.get("status", "empty"),
            "fill_probability": float(settings.fill_quality_default_probability or 0.95),
            "fill_slippage_penalty_bps": 0.0,
            "fill_delay_penalty_bps": 0.0,
            "total_fill_quality_penalty_bps": 0.0,
            "approved": True,
            "message": "No fill-quality stats; neutral adjustment",
        }

    sample_count = int(stats.get("sample_count") or 0)
    min_samples = int(settings.fill_quality_min_samples or 5)

    if sample_count < min_samples:
        return {
            "status": "collecting",
            "sample_count": sample_count,
            "fill_probability": float(settings.fill_quality_default_probability or 0.95),
            "fill_slippage_penalty_bps": 0.0,
            "fill_delay_penalty_bps": 0.0,
            "total_fill_quality_penalty_bps": 0.0,
            "approved": True,
            "message": "Not enough fill-quality samples; neutral adjustment",
        }

    fill_ratio = float(stats.get("ewma_fill_ratio") or 1.0)
    fill_probability = max(
        float(settings.fill_quality_min_probability or 0.70),
        min(1.0, fill_ratio),
    )

    ewma_slippage = float(stats.get("ewma_slippage_bps") or 0.0)
    bad_slippage = float(settings.fill_quality_bad_slippage_bps or 35.0)

    slippage_penalty = max(0.0, ewma_slippage - bad_slippage)

    if execution_mode == "paper":
        slippage_penalty = min(
            slippage_penalty,
            float(settings.fill_quality_paper_max_slippage_penalty_bps or 20.0),
        )
    else:
        slippage_penalty = min(
            slippage_penalty,
            float(settings.fill_quality_live_max_slippage_penalty_bps or 50.0),
        )

    delay = float(stats.get("ewma_fill_delay_seconds") or 0.0)
    delay_penalty = 0.0
    if delay > float(settings.fill_quality_delay_penalty_threshold_seconds or 30.0):
        delay_penalty = float(settings.fill_quality_delay_penalty_bps or 5.0)

    total_penalty = slippage_penalty + delay_penalty

    approved = True
    message = "Fill quality acceptable"

    if execution_mode != "paper" and bool(settings.fill_quality_live_block_bad_fills):
        if ewma_slippage > bad_slippage * 2:
            approved = False
            message = f"EWMA slippage {ewma_slippage:.2f}bps too high"
        elif fill_ratio < float(settings.fill_quality_bad_fill_ratio or 0.70):
            approved = False
            message = f"EWMA fill ratio {fill_ratio:.2f} too low"

    return {
        "status": "ready",
        "sample_count": sample_count,
        "fill_probability": round(fill_probability, 6),
        "ewma_slippage_bps": round(ewma_slippage, 4),
        "ewma_fill_ratio": round(fill_ratio, 6),
        "ewma_fill_delay_seconds": round(delay, 4),
        "fill_slippage_penalty_bps": round(slippage_penalty, 4),
        "fill_delay_penalty_bps": round(delay_penalty, 4),
        "total_fill_quality_penalty_bps": round(total_penalty, 4),
        "approved": approved,
        "message": message,
    }
