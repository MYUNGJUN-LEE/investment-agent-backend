from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Any

from app.brokers.kis_client import (
    KisApiError,
    KisClient,
    KisConfigError,
    is_invalid_account_error,
)
from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS broker_balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    broker TEXT NOT NULL,
    account_no TEXT,
    total_cash REAL,
    total_value REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS broker_positions (
    broker TEXT NOT NULL,
    account_no TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    name TEXT,
    quantity INTEGER NOT NULL DEFAULT 0,
    avg_price REAL,
    current_price REAL,
    eval_amount REAL,
    pnl REAL,
    opened_at TEXT,
    synced_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (broker, account_no, symbol)
);

CREATE TABLE IF NOT EXISTS broker_order_executions (
    broker TEXT NOT NULL,
    account_no TEXT NOT NULL DEFAULT '',
    order_no TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    side TEXT,
    order_qty INTEGER,
    filled_qty INTEGER,
    remaining_qty INTEGER,
    order_price REAL,
    avg_fill_price REAL,
    order_time TEXT,
    status TEXT,
    synced_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (broker, account_no, order_no, symbol)
);
"""


def sync_kis_account(
    client: KisClient | None = None,
    db_path: Path | str | None = None,
    lookback_days: int = 1,
) -> dict[str, Any]:
    """Synchronize KIS balance, positions, and recent order executions."""
    client = client or KisClient()
    try:
        balance = client.get_balance()
        end = datetime.now()
        start = end - timedelta(days=max(0, lookback_days))
        executions = client.get_daily_order_executions(
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
    except KisConfigError as exc:
        return _kis_config_error_result(client=client, exc=exc)
    except KisApiError as exc:
        if is_invalid_account_error(exc):
            return _kis_config_error_result(client=client, exc=exc)
        raise
    return record_kis_sync(
        balance=balance,
        executions=executions,
        account_no=client.account_no or "",
        db_path=db_path,
    )


def record_kis_sync(
    balance: dict[str, Any],
    executions: dict[str, Any],
    account_no: str = "",
    db_path: Path | str | None = None,
    complete_snapshot: bool | None = None,
) -> dict[str, Any]:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    positions = _parse_positions(balance)
    orders = _parse_executions(executions)
    totals = _parse_balance_totals(balance)

    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_column(conn, "broker_positions", "opened_at", "TEXT")
        conn.execute(
            """
            INSERT INTO broker_balance_snapshots (
                created_at, broker, account_no, total_cash, total_value, raw_json
            )
            VALUES (?, 'KIS', ?, ?, ?, ?)
            """,
            (
                now,
                account_no,
                totals.get("total_cash"),
                totals.get("total_value"),
                _json(balance),
            ),
        )
        conn.executemany(
            """
            INSERT INTO broker_positions (
                broker, account_no, symbol, name, quantity, avg_price,
                current_price, eval_amount, pnl, opened_at, synced_at, raw_json
            )
            VALUES ('KIS', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(broker, account_no, symbol) DO UPDATE SET
                name = excluded.name,
                quantity = excluded.quantity,
                avg_price = excluded.avg_price,
                current_price = excluded.current_price,
                eval_amount = excluded.eval_amount,
                pnl = excluded.pnl,
                opened_at = COALESCE(broker_positions.opened_at, excluded.opened_at),
                synced_at = excluded.synced_at,
                raw_json = excluded.raw_json
            """,
            [
                (
                    account_no,
                    position["symbol"],
                    position.get("name"),
                    position.get("quantity", 0),
                    position.get("avg_price"),
                    position.get("current_price"),
                    position.get("eval_amount"),
                    position.get("pnl"),
                    now,
                    now,
                    _json(position.get("raw", position)),
                )
                for position in positions
            ],
        )
        close_missing_positions = (
            bool(balance.get("__complete_snapshot"))
            if complete_snapshot is None
            else complete_snapshot
        )
        closed_position_count = (
            _mark_missing_positions_closed(
                conn=conn,
                account_no=account_no,
                current_symbols=[position["symbol"] for position in positions],
                synced_at=now,
            )
            if close_missing_positions
            else 0
        )
        conn.executemany(
            """
            INSERT INTO broker_order_executions (
                broker, account_no, order_no, symbol, side, order_qty,
                filled_qty, remaining_qty, order_price, avg_fill_price,
                order_time, status, synced_at, raw_json
            )
            VALUES ('KIS', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(broker, account_no, order_no, symbol) DO UPDATE SET
                side = excluded.side,
                order_qty = excluded.order_qty,
                filled_qty = excluded.filled_qty,
                remaining_qty = excluded.remaining_qty,
                order_price = excluded.order_price,
                avg_fill_price = excluded.avg_fill_price,
                order_time = excluded.order_time,
                status = excluded.status,
                synced_at = excluded.synced_at,
                raw_json = excluded.raw_json
            """,
            [
                (
                    account_no,
                    order.get("order_no") or "",
                    order.get("symbol") or "",
                    order.get("side"),
                    order.get("order_qty"),
                    order.get("filled_qty"),
                    order.get("remaining_qty"),
                    order.get("order_price"),
                    order.get("avg_fill_price"),
                    order.get("order_time"),
                    order.get("status"),
                    now,
                    _json(order.get("raw", order)),
                )
                for order in orders
                if order.get("order_no")
            ],
        )

    return {
        "status": "success",
        "broker": "KIS",
        "account_no": account_no,
        "position_count": len(positions),
        "closed_position_count": closed_position_count,
        "complete_snapshot": close_missing_positions,
        "execution_count": len(orders),
        "total_cash": totals.get("total_cash"),
        "total_value": totals.get("total_value"),
        "synced_at": now,
    }


def _kis_config_error_result(
    client: KisClient,
    exc: Exception,
) -> dict[str, Any]:
    error_code = getattr(exc, "error_code", None)
    error_description = getattr(exc, "error_description", None)
    return {
        "status": "config_error",
        "broker": "KIS",
        "message": (
            "KIS account configuration is invalid. Check that "
            "KIS_ACCOUNT_NO is the 8-digit CANO only, "
            "KIS_ACCOUNT_PRODUCT_CODE is the 2-digit product code, and the "
            "account belongs to the configured paper/live KIS app key."
        ),
        "detail": str(exc),
        "kis_error_code": error_code,
        "kis_error_description": error_description,
        "is_paper": client.is_paper,
        "account_no_configured": bool(client.account_no),
        "account_no_length": len(client.account_no or ""),
        "account_no_last4": (client.account_no or "")[-4:],
        "account_product_code": client.account_product_code or "",
        "synced_at": _now(),
    }


def _parse_balance_totals(data: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(data, "output2")
    row = rows[0] if rows else {}
    return {
        "total_cash": _pick_float(
            row,
            "dnca_tot_amt",
            "nass_amt",
            "tot_evlu_amt",
        ),
        "total_value": _pick_float(
            row,
            "tot_evlu_amt",
            "scts_evlu_amt",
            "evlu_amt_smtl_amt",
        ),
    }


def _mark_missing_positions_closed(
    conn: sqlite3.Connection,
    account_no: str,
    current_symbols: list[str],
    synced_at: str,
) -> int:
    raw = _json(
        {
            "source": "record_kis_sync",
            "message": "Position was absent from the latest broker balance.",
        }
    )
    if current_symbols:
        placeholders = ",".join("?" for _ in current_symbols)
        cursor = conn.execute(
            f"""
            UPDATE broker_positions
            SET quantity = 0,
                eval_amount = 0,
                pnl = 0,
                synced_at = ?,
                raw_json = ?
            WHERE broker = 'KIS'
              AND account_no = ?
              AND quantity > 0
              AND symbol NOT IN ({placeholders})
            """,
            (synced_at, raw, account_no, *current_symbols),
        )
    else:
        cursor = conn.execute(
            """
            UPDATE broker_positions
            SET quantity = 0,
                eval_amount = 0,
                pnl = 0,
                synced_at = ?,
                raw_json = ?
            WHERE broker = 'KIS'
              AND account_no = ?
              AND quantity > 0
            """,
            (synced_at, raw, account_no),
        )
    return int(cursor.rowcount or 0)


def _parse_positions(data: dict[str, Any]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for row in _rows(data, "output1"):
        symbol = str(row.get("pdno") or row.get("prdt_code") or "").strip()
        quantity = _pick_int(row, "hldg_qty", "ord_psbl_qty", "qty")
        if not symbol or not quantity:
            continue
        positions.append(
            {
                "symbol": symbol,
                "name": row.get("prdt_name") or row.get("prdt_name120"),
                "quantity": quantity,
                "avg_price": _pick_float(row, "pchs_avg_pric", "avg_prvs"),
                "current_price": _pick_float(row, "prpr", "now_pric"),
                "eval_amount": _pick_float(row, "evlu_amt", "evlu_pfls_amt"),
                "pnl": _pick_float(row, "evlu_pfls_amt", "evlu_erng_rt"),
                "raw": row,
            }
        )
    return positions


def _parse_executions(data: dict[str, Any]) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for row in _rows(data, "output1", "output"):
        order_no = str(row.get("odno") or row.get("ODNO") or "").strip()
        symbol = str(row.get("pdno") or row.get("PDNO") or "").strip()
        filled_qty = _pick_int(row, "tot_ccld_qty", "ccld_qty")
        order_qty = _pick_int(row, "ord_qty")
        remaining_qty = _pick_int(row, "rmn_qty")
        orders.append(
            {
                "order_no": order_no,
                "symbol": symbol,
                "side": row.get("sll_buy_dvsn_cd_name")
                or row.get("sll_buy_dvsn_cd")
                or row.get("ord_dvsn_name"),
                "order_qty": order_qty,
                "filled_qty": filled_qty,
                "remaining_qty": remaining_qty,
                "order_price": _pick_float(row, "ord_unpr"),
                "avg_fill_price": _pick_float(row, "avg_prvs", "tot_ccld_amt"),
                "order_time": row.get("ord_tmd") or row.get("ord_dt"),
                "status": _execution_status(order_qty, filled_qty, remaining_qty),
                "raw": row,
            }
        )
    return orders


def _execution_status(
    order_qty: int | None,
    filled_qty: int | None,
    remaining_qty: int | None,
) -> str:
    if remaining_qty is not None and remaining_qty > 0:
        return "PARTIALLY_FILLED" if filled_qty else "OPEN"
    if filled_qty and order_qty and filled_qty >= order_qty:
        return "FILLED"
    if filled_qty:
        return "PARTIALLY_FILLED"
    return "UNKNOWN"


def _rows(data: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows.append(value)
    return rows


def _pick_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _pick_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _to_int(row.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.broker_sync_db_path)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
