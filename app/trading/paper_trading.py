from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3
from typing import Any

from app.config import settings
from app.models import PaperRunRequest
from app.storage.market_data import get_latest_market_context
from app.trading import cost_model, performance, risk_manager


DEFAULT_DB_PATH = Path("data/paper_trading.sqlite3")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    market TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER,
    confidence REAL NOT NULL,
    reason TEXT,
    source TEXT NOT NULL,
    signal_time TEXT,
    decision_price REAL,
    signal_score REAL,
    stop_loss REAL,
    take_profit REAL,
    market_regime TEXT,
    model_version TEXT,
    sector TEXT,
    raw_payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    effective_price REAL,
    order_price REAL,
    fill_price REAL,
    requested_quantity INTEGER,
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    remaining_quantity INTEGER NOT NULL DEFAULT 0,
    amount REAL NOT NULL,
    requested_amount REAL,
    slippage_bps REAL,
    signal_time TEXT,
    decision_price REAL,
    signal_score REAL,
    position_size REAL,
    stop_loss REAL,
    take_profit REAL,
    market_regime TEXT,
    reason TEXT,
    model_version TEXT,
    sector TEXT,
    order_state TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    cancel_status TEXT,
    last_error TEXT,
    total_cost REAL NOT NULL DEFAULT 0,
    commission REAL NOT NULL DEFAULT 0,
    tax REAL NOT NULL DEFAULT 0,
    spread_cost REAL NOT NULL DEFAULT 0,
    slippage_cost REAL NOT NULL DEFAULT 0,
    fx_cost REAL NOT NULL DEFAULT 0,
    financing_cost REAL NOT NULL DEFAULT 0,
    borrow_cost REAL NOT NULL DEFAULT 0,
    gross_realized_pnl REAL NOT NULL DEFAULT 0,
    net_realized_pnl REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    market TEXT NOT NULL,
    sector TEXT,
    quantity INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    cost_basis REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_previews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    preview_token TEXT,
    symbol TEXT NOT NULL,
    name TEXT,
    market TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    signal_type TEXT,
    side TEXT,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    amount REAL NOT NULL,
    confidence REAL NOT NULL,
    final_grade TEXT,
    strategy_message TEXT NOT NULL,
    risk_approved INTEGER NOT NULL DEFAULT 0,
    risk_code TEXT,
    risk_message TEXT,
    raw_pipeline_result TEXT NOT NULL,
    raw_strategy_decision TEXT NOT NULL,
    confirmed_at TEXT,
    paper_order_id INTEGER
);
"""


def run_paper_once(
    req: PaperRunRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Record one paper-trading signal and virtual fill."""
    req = _with_market_context_defaults(req)
    resolved_db_path = settings.storage_path(db_path or DEFAULT_DB_PATH)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    now = _now()
    with _connect(resolved_db_path) as conn:
        initialize_db(conn)
        signal_id = _insert_signal(conn, req, now)

        if req.signal_type == "entry":
            order = _handle_entry(conn, req, signal_id, now)
        else:
            order = _handle_exit(conn, req, signal_id, now)

        position = _get_position(conn, req.symbol)
        performance_metrics = performance.record_performance_snapshot(
            conn=conn,
            created_at=now,
            symbol=req.symbol,
            market=req.market,
            strategy_type=req.strategy_type,
            portfolio_value=risk_manager.DEFAULT_LIMITS.portfolio_value,
            beta=req.market_beta,
        )

    return {
        "status": "success" if order["status"] in ("FILLED", "PARTIALLY_FILLED") else "rejected",
        "signal_id": signal_id,
        "order_id": order["id"],
        "order_status": order["status"],
        "symbol": req.symbol,
        "side": order["side"],
        "price": req.price,
        "quantity": order["quantity"],
        "amount": order["amount"],
        "effective_price": order.get("effective_price"),
        "total_cost": order.get("total_cost", 0),
        "realized_pnl": order.get("realized_pnl", 0),
        "cost_breakdown": order.get("cost_breakdown"),
        "performance_metrics": performance_metrics,
        "message": order["message"],
        "position": position,
    }


def initialize_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    performance.initialize_performance_db(conn)
    _ensure_signal_columns(conn)
    _ensure_order_columns(conn)
    _ensure_position_columns(conn)
    conn.execute(
        """
        UPDATE positions
        SET cost_basis = avg_price * quantity
        WHERE quantity > 0 AND (cost_basis IS NULL OR cost_basis = 0)
        """
    )


def get_paper_account_snapshot(
    db_path: Path | str | None = None,
    account_equity: float | None = None,
) -> dict[str, Any]:
    """Return simulated account equity, invested capital, and available cash."""
    equity = float(account_equity or risk_manager.DEFAULT_LIMITS.portfolio_value)
    resolved_db_path = settings.storage_path(db_path or DEFAULT_DB_PATH)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(resolved_db_path) as conn:
        initialize_db(conn)
        row = conn.execute(
            """
            SELECT COALESCE(SUM(cost_basis), 0) AS invested_amount
            FROM positions
            WHERE quantity > 0
            """
        ).fetchone()
    invested_amount = float(row["invested_amount"] or 0)
    cash_available = max(0.0, equity - invested_amount)
    return {
        "mode": "paper",
        "account_equity": equity,
        "invested_amount": round(invested_amount, 2),
        "cash_available": round(cash_available, 2),
    }


def _ensure_order_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "paper_orders", "realized_pnl", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "effective_price", "REAL")
    _ensure_column(conn, "paper_orders", "order_price", "REAL")
    _ensure_column(conn, "paper_orders", "fill_price", "REAL")
    _ensure_column(conn, "paper_orders", "requested_quantity", "INTEGER")
    _ensure_column(conn, "paper_orders", "filled_quantity", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "remaining_quantity", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "requested_amount", "REAL")
    _ensure_column(conn, "paper_orders", "slippage_bps", "REAL")
    _ensure_column(conn, "paper_orders", "signal_time", "TEXT")
    _ensure_column(conn, "paper_orders", "decision_price", "REAL")
    _ensure_column(conn, "paper_orders", "signal_score", "REAL")
    _ensure_column(conn, "paper_orders", "position_size", "REAL")
    _ensure_column(conn, "paper_orders", "stop_loss", "REAL")
    _ensure_column(conn, "paper_orders", "take_profit", "REAL")
    _ensure_column(conn, "paper_orders", "market_regime", "TEXT")
    _ensure_column(conn, "paper_orders", "reason", "TEXT")
    _ensure_column(conn, "paper_orders", "model_version", "TEXT")
    _ensure_column(conn, "paper_orders", "sector", "TEXT")
    _ensure_column(conn, "paper_orders", "order_state", "TEXT")
    _ensure_column(conn, "paper_orders", "retry_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "cancel_status", "TEXT")
    _ensure_column(conn, "paper_orders", "last_error", "TEXT")
    _ensure_column(conn, "paper_orders", "total_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "commission", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "tax", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "spread_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "slippage_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "fx_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "financing_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "borrow_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "gross_realized_pnl", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "paper_orders", "net_realized_pnl", "REAL NOT NULL DEFAULT 0")


def _ensure_position_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "positions", "cost_basis", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "positions", "sector", "TEXT")


def _ensure_signal_columns(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "signals", "signal_time", "TEXT")
    _ensure_column(conn, "signals", "decision_price", "REAL")
    _ensure_column(conn, "signals", "signal_score", "REAL")
    _ensure_column(conn, "signals", "stop_loss", "REAL")
    _ensure_column(conn, "signals", "take_profit", "REAL")
    _ensure_column(conn, "signals", "market_regime", "TEXT")
    _ensure_column(conn, "signals", "model_version", "TEXT")
    _ensure_column(conn, "signals", "sector", "TEXT")


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_signal(conn: sqlite3.Connection, req: PaperRunRequest, created_at: str) -> int:
    payload = req.model_dump()
    cursor = conn.execute(
        """
        INSERT INTO signals (
            created_at, symbol, name, market, strategy_type, signal_type,
            price, quantity, confidence, reason, source, signal_time,
            decision_price, signal_score, stop_loss, take_profit, market_regime,
            model_version, sector, raw_payload
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            req.symbol,
            req.name,
            req.market,
            req.strategy_type,
            req.signal_type,
            req.price,
            req.quantity,
            req.confidence,
            req.reason,
            req.source,
            req.signal_time or created_at,
            req.decision_price,
            req.signal_score,
            req.stop_loss,
            req.take_profit,
            req.market_regime,
            req.model_version,
            req.sector,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    return int(cursor.lastrowid)


def _handle_entry(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    signal_id: int,
    created_at: str,
) -> dict[str, Any]:
    if not req.quantity:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="BUY",
            price=req.price,
            quantity=0,
            status="REJECTED",
            message="Entry signal requires a positive quantity",
            req=req,
        )

    decision = risk_manager.approve_order(
        conn=conn,
        req=req,
        side="BUY",
        quantity=req.quantity,
        now=created_at,
    )
    if not decision.approved:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="BUY",
            price=req.price,
            quantity=0,
            status="REJECTED",
            message=f"Risk rejected: {decision.message} ({decision.code})",
            req=req,
        )

    requested_qty = req.quantity
    filled_qty = _resolved_fill_quantity(req, requested_qty)
    if filled_qty <= 0:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="BUY",
            price=req.price,
            quantity=0,
            requested_quantity=requested_qty,
            status="REJECTED",
            message="Order was not filled",
            req=req,
        )

    execution = cost_model.estimate_order_cost(req=req, side="BUY", quantity=filled_qty)
    current = _get_position(conn, req.symbol)
    current_qty = int(current["quantity"]) if current else 0
    current_cost_basis = _position_cost_basis(current) if current else 0.0
    realized_pnl = float(current["realized_pnl"]) if current else 0.0

    new_qty = current_qty + filled_qty
    new_cost_basis = current_cost_basis + execution.requested_amount + execution.total_cost
    new_avg = new_cost_basis / new_qty

    _upsert_position(
        conn=conn,
        symbol=req.symbol,
        name=req.name,
        market=req.market,
        sector=req.sector,
        quantity=new_qty,
        avg_price=new_avg,
        cost_basis=new_cost_basis,
        realized_pnl=realized_pnl,
        updated_at=created_at,
    )
    return _insert_order(
        conn=conn,
        signal_id=signal_id,
        created_at=created_at,
        symbol=req.symbol,
        side="BUY",
        price=req.price,
        effective_price=execution.effective_price,
        quantity=filled_qty,
        requested_quantity=requested_qty,
        status="FILLED" if filled_qty == requested_qty else "PARTIALLY_FILLED",
        message="Paper entry filled",
        cost=execution,
        req=req,
    )


def _handle_exit(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    signal_id: int,
    created_at: str,
) -> dict[str, Any]:
    current = _get_position(conn, req.symbol)
    if not current or int(current["quantity"]) <= 0:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="SELL",
            price=req.price,
            quantity=0,
            status="REJECTED",
            message="No open paper position to exit",
            req=req,
        )

    current_qty = int(current["quantity"])
    exit_qty = req.quantity or current_qty
    if exit_qty > current_qty:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="SELL",
            price=req.price,
            quantity=0,
            status="REJECTED",
            message="Exit quantity exceeds open paper position",
            req=req,
        )

    avg_price = float(current["avg_price"])
    decision = risk_manager.approve_order(
        conn=conn,
        req=req,
        side="SELL",
        quantity=exit_qty,
        now=created_at,
    )
    if not decision.approved:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="SELL",
            price=req.price,
            quantity=0,
            status="REJECTED",
            message=f"Risk rejected: {decision.message} ({decision.code})",
            req=req,
        )

    requested_qty = exit_qty
    filled_qty = _resolved_fill_quantity(req, requested_qty)
    if filled_qty <= 0:
        return _insert_order(
            conn=conn,
            signal_id=signal_id,
            created_at=created_at,
            symbol=req.symbol,
            side="SELL",
            price=req.price,
            quantity=0,
            requested_quantity=requested_qty,
            status="REJECTED",
            message="Order was not filled",
            req=req,
        )

    execution = cost_model.estimate_order_cost(req=req, side="SELL", quantity=filled_qty)
    current_cost_basis = _position_cost_basis(current)
    allocated_cost_basis = current_cost_basis * filled_qty / current_qty
    gross_realized_pnl = (execution.effective_price - avg_price) * filled_qty
    net_proceeds = execution.requested_amount - execution.total_cost
    order_realized_pnl = net_proceeds - allocated_cost_basis
    realized_pnl = float(current["realized_pnl"]) + order_realized_pnl
    remaining_qty = current_qty - filled_qty
    remaining_cost_basis = current_cost_basis - allocated_cost_basis
    remaining_avg = remaining_cost_basis / remaining_qty if remaining_qty > 0 else 0.0

    _upsert_position(
        conn=conn,
        symbol=req.symbol,
        name=req.name or current["name"],
        market=req.market,
        sector=req.sector or current.get("sector"),
        quantity=remaining_qty,
        avg_price=remaining_avg,
        cost_basis=max(0.0, remaining_cost_basis) if remaining_qty > 0 else 0.0,
        realized_pnl=realized_pnl,
        updated_at=created_at,
    )
    return _insert_order(
        conn=conn,
        signal_id=signal_id,
        created_at=created_at,
        symbol=req.symbol,
        side="SELL",
        price=req.price,
        effective_price=execution.effective_price,
        quantity=filled_qty,
        requested_quantity=requested_qty,
        status="FILLED" if filled_qty == requested_qty else "PARTIALLY_FILLED",
        message="Paper exit filled",
        realized_pnl=order_realized_pnl,
        gross_realized_pnl=gross_realized_pnl,
        net_realized_pnl=order_realized_pnl,
        cost=execution,
        req=req,
    )


def _insert_order(
    conn: sqlite3.Connection,
    signal_id: int,
    created_at: str,
    symbol: str,
    side: str,
    price: float,
    quantity: int,
    status: str,
    message: str,
    effective_price: float | None = None,
    requested_quantity: int | None = None,
    realized_pnl: float = 0.0,
    gross_realized_pnl: float = 0.0,
    net_realized_pnl: float | None = None,
    cost: cost_model.OrderCost | None = None,
    req: PaperRunRequest | None = None,
) -> dict[str, Any]:
    amount = round((cost.effective_amount if cost else price * quantity), 2)
    requested_amount = round((cost.requested_amount if cost else price * quantity), 2)
    total_cost = round(cost.total_cost, 2) if cost else 0.0
    net_realized_pnl = realized_pnl if net_realized_pnl is None else net_realized_pnl
    requested_quantity = quantity if requested_quantity is None else requested_quantity
    remaining_quantity = max(0, requested_quantity - quantity)
    order_price = req.order_price if req and req.order_price else price
    fill_price = req.fill_price if req and req.fill_price else (effective_price if effective_price is not None else price)
    slippage_bps = _slippage_bps(side, order_price, fill_price)
    signal_time = req.signal_time if req and req.signal_time else created_at
    decision_price = req.decision_price if req and req.decision_price else price
    position_size = _position_size(req, amount) if req else None
    cursor = conn.execute(
        """
        INSERT INTO paper_orders (
            signal_id, created_at, symbol, side, price, effective_price,
            order_price, fill_price, requested_quantity, quantity, filled_quantity,
            remaining_quantity, amount, requested_amount, slippage_bps, signal_time,
            decision_price, signal_score, position_size, stop_loss, take_profit,
            market_regime, reason, model_version, sector, order_state, retry_count,
            cancel_status, last_error, total_cost, commission, tax,
            spread_cost, slippage_cost, fx_cost, financing_cost, borrow_cost,
            gross_realized_pnl, net_realized_pnl, realized_pnl, status, message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_id,
            created_at,
            symbol,
            side,
            price,
            effective_price if effective_price is not None else price,
            order_price,
            fill_price,
            requested_quantity,
            quantity,
            quantity if status in ("FILLED", "PARTIALLY_FILLED") else 0,
            remaining_quantity if status == "PARTIALLY_FILLED" else 0,
            amount,
            requested_amount,
            slippage_bps,
            signal_time,
            decision_price,
            req.signal_score if req else None,
            position_size,
            req.stop_loss if req else None,
            req.take_profit if req else None,
            req.market_regime if req else None,
            req.reason if req else None,
            req.model_version if req else None,
            req.sector if req else None,
            status,
            0,
            None,
            None,
            total_cost,
            round(cost.commission, 2) if cost else 0.0,
            round(cost.tax, 2) if cost else 0.0,
            round(cost.spread_cost, 2) if cost else 0.0,
            round(cost.slippage_cost, 2) if cost else 0.0,
            round(cost.fx_cost, 2) if cost else 0.0,
            round(cost.financing_cost, 2) if cost else 0.0,
            round(cost.borrow_cost, 2) if cost else 0.0,
            round(gross_realized_pnl, 2),
            round(net_realized_pnl, 2),
            round(realized_pnl, 2),
            status,
            message,
        ),
    )
    cost_breakdown = cost.to_dict() if cost else None
    return {
        "id": int(cursor.lastrowid),
        "side": side,
        "quantity": quantity,
        "amount": amount,
        "effective_price": round(effective_price if effective_price is not None else price, 6),
        "fill_price": round(fill_price, 6),
        "slippage_bps": slippage_bps,
        "total_cost": total_cost,
        "realized_pnl": round(realized_pnl, 2),
        "gross_realized_pnl": round(gross_realized_pnl, 2),
        "net_realized_pnl": round(net_realized_pnl, 2),
        "cost_breakdown": cost_breakdown,
        "status": status,
        "message": message,
    }


def _upsert_position(
    conn: sqlite3.Connection,
    symbol: str,
    name: str | None,
    market: str,
    sector: str | None,
    quantity: int,
    avg_price: float,
    cost_basis: float,
    realized_pnl: float,
    updated_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO positions (
            symbol, name, market, sector, quantity, avg_price, cost_basis, realized_pnl, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            name = excluded.name,
            market = excluded.market,
            sector = excluded.sector,
            quantity = excluded.quantity,
            avg_price = excluded.avg_price,
            cost_basis = excluded.cost_basis,
            realized_pnl = excluded.realized_pnl,
            updated_at = excluded.updated_at
        """,
        (
            symbol,
            name,
            market,
            sector,
            quantity,
            round(avg_price, 4),
            round(cost_basis, 4),
            round(realized_pnl, 2),
            updated_at,
        ),
    )


def _get_position(conn: sqlite3.Connection, symbol: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT symbol, name, market, sector, quantity, avg_price, cost_basis, realized_pnl, updated_at
        FROM positions
        WHERE symbol = ?
        """,
        (symbol,),
    ).fetchone()
    return dict(row) if row else None


def _position_cost_basis(position: dict[str, Any] | sqlite3.Row | None) -> float:
    if not position:
        return 0.0
    cost_basis = float(position["cost_basis"] or 0)
    if cost_basis > 0:
        return cost_basis
    return float(position["avg_price"] or 0) * int(position["quantity"] or 0)


def _resolved_fill_quantity(req: PaperRunRequest, requested_quantity: int) -> int:
    if req.filled_quantity is None:
        return requested_quantity
    return max(0, min(int(req.filled_quantity), requested_quantity))


def _slippage_bps(side: str, order_price: float, fill_price: float) -> float | None:
    if order_price <= 0:
        return None
    signed = (fill_price - order_price) / order_price * 10_000
    adverse = signed if side == "BUY" else -signed
    return round(adverse, 4)


def _position_size(req: PaperRunRequest, amount: float) -> float | None:
    if req.position_size is not None:
        return req.position_size
    equity = req.account_equity or risk_manager.DEFAULT_LIMITS.portfolio_value
    if equity <= 0:
        return None
    return round(amount / equity, 6)


def _with_market_context_defaults(req: PaperRunRequest) -> PaperRunRequest:
    if req.market_regime:
        return req
    latest_context = get_latest_market_context()
    if not latest_context or not latest_context.get("market_regime"):
        return req
    return req.model_copy(update={"market_regime": latest_context["market_regime"]})


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
