from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS outcome_attribution_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    execution_mode TEXT,
    entry_time TEXT,
    exit_time TEXT,
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    gross_return_bps REAL,
    realized_return_bps REAL,
    realized_risk_cost_bps REAL,
    trading_cost_bps REAL,
    slippage_cost_bps REAL,
    realized_net_edge_bps REAL,
    predicted_net_edge_bps REAL,
    final_entry_edge_bps REAL,
    signal_component_bps REAL,
    market_regime_component_bps REAL,
    execution_component_bps REAL,
    sizing_component_bps REAL,
    time_decay_component_bps REAL,
    unexplained_component_bps REAL,
    outcome_label TEXT,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcome_attr_symbol_time
ON outcome_attribution_events(symbol, created_at);

CREATE INDEX IF NOT EXISTS idx_outcome_attr_label_time
ON outcome_attribution_events(outcome_label, created_at);
"""


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _db_path(db_path: Path | str | None = None) -> Path:
    return settings.storage_path(db_path or settings.outcome_attribution_db_path)


def initialize_outcome_attribution_db(db_path: Path | str | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip()


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _bps_return(entry_price: float | None, exit_price: float | None) -> float | None:
    if entry_price is None or exit_price is None:
        return None
    if entry_price <= 0 or exit_price <= 0:
        return None
    return (exit_price / entry_price - 1.0) * 10_000.0


def _holding_seconds(entry_time: Any, exit_time: Any) -> float | None:
    start = _parse_time(entry_time)
    end = _parse_time(exit_time)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _default_trading_cost_bps() -> float:
    commission = max(0.0, float(settings.commission_rate or 0.0))
    sell_tax = max(0.0, float(settings.kr_stock_sell_tax_rate or 0.0))
    return (commission * 2.0 + sell_tax) * 10_000.0


def outcome_attribution_from_trade(trade: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _normalize_symbol(trade.get("symbol"))
    if not symbol:
        return None

    entry_price = _first_float(
        trade.get("entry_price"),
        trade.get("buy_price"),
        trade.get("filled_entry_price"),
        trade.get("avg_entry_price"),
    )
    exit_price = _first_float(
        trade.get("exit_price"),
        trade.get("sell_price"),
        trade.get("filled_exit_price"),
        trade.get("avg_exit_price"),
        trade.get("filled_price"),
    )

    gross_return_bps = _to_float(trade.get("gross_return_bps"))
    if gross_return_bps is None:
        gross_return_bps = _bps_return(entry_price, exit_price)

    if gross_return_bps is None:
        return None

    hold_seconds = _holding_seconds(
        trade.get("entry_time") or trade.get("opened_at"),
        trade.get("exit_time") or trade.get("closed_at"),
    )
    min_hold = int(settings.outcome_attribution_min_hold_seconds or 60)
    if hold_seconds is not None and hold_seconds < min_hold:
        return None

    trading_cost_bps = _first_float(
        trade.get("trading_cost_bps"),
        trade.get("trading_cost"),
    )
    if trading_cost_bps is None:
        trading_cost_bps = _default_trading_cost_bps()

    slippage_cost_bps = _first_float(
        trade.get("slippage_cost_bps"),
        trade.get("slippage_cost"),
    )
    if slippage_cost_bps is None:
        slippage_cost_bps = 0.0

    realized_risk_bps = _first_float(
        trade.get("realized_risk_bps"),
        trade.get("max_adverse_excursion_bps"),
    )
    if realized_risk_bps is None:
        realized_risk_bps = 0.0

    risk_weight = float(settings.outcome_attribution_risk_weight or 0.10)
    realized_risk_cost_bps = realized_risk_bps * risk_weight

    realized_return_bps = _to_float(trade.get("realized_return_bps"))
    if realized_return_bps is None:
        realized_return_bps = gross_return_bps

    realized_net_edge_bps = (
        realized_return_bps
        - realized_risk_cost_bps
        - trading_cost_bps
        - slippage_cost_bps
    )

    predicted_net_edge = _first_float(
        trade.get("final_entry_edge"),
        trade.get("fill_quality_adjusted_edge"),
        trade.get("portfolio_adjusted_net_edge"),
        trade.get("net_edge"),
    )
    if predicted_net_edge is None:
        predicted_net_edge = 0.0

    final_entry_edge = _first_float(trade.get("final_entry_edge"), predicted_net_edge)
    if final_entry_edge is None:
        final_entry_edge = predicted_net_edge

    regime_gate = _safe_dict(trade.get("regime_gate"))
    portfolio_penalty = _safe_dict(trade.get("portfolio_penalty"))
    fill_quality = _safe_dict(trade.get("fill_quality"))
    signal_decay = _safe_dict(trade.get("signal_decay"))
    sizing = _safe_dict(trade.get("position_sizing"))

    # Components are approximate explanatory attribution, not causal proof.
    signal_component = predicted_net_edge

    market_adjustment = _to_float(regime_gate.get("hurdle_adjustment_bps")) or 0.0
    market_regime_component = (
        -abs(market_adjustment) if market_adjustment > 0 else abs(market_adjustment)
    )

    execution_component = -(
        abs(_to_float(fill_quality.get("total_fill_quality_penalty_bps")) or 0.0)
        + abs(slippage_cost_bps)
    )

    portfolio_component = -abs(
        _to_float(portfolio_penalty.get("total_penalty_bps")) or 0.0
    )

    time_decay_component = -abs(
        _to_float(signal_decay.get("signal_decay_penalty_bps")) or 0.0
    )

    edge_multiplier = _to_float(sizing.get("edge_position_multiplier"))
    sizing_component = 0.0
    if edge_multiplier is not None and edge_multiplier > 1.0:
        sizing_component = -abs((edge_multiplier - 1.0) * 10.0)

    explained = (
        signal_component
        + market_regime_component
        + execution_component
        + portfolio_component
        + time_decay_component
        + sizing_component
    )

    unexplained = realized_net_edge_bps - explained

    if realized_net_edge_bps > 0:
        label = "win"
    elif realized_net_edge_bps < 0:
        label = "loss"
    else:
        label = "flat"

    return {
        "created_at": _now(),
        "symbol": symbol,
        "execution_mode": str(trade.get("execution_mode") or ""),
        "entry_time": str(trade.get("entry_time") or trade.get("opened_at") or ""),
        "exit_time": str(trade.get("exit_time") or trade.get("closed_at") or ""),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": _to_float(trade.get("quantity")),
        "gross_return_bps": gross_return_bps,
        "realized_return_bps": realized_return_bps,
        "realized_risk_cost_bps": realized_risk_cost_bps,
        "trading_cost_bps": trading_cost_bps,
        "slippage_cost_bps": slippage_cost_bps,
        "realized_net_edge_bps": realized_net_edge_bps,
        "predicted_net_edge_bps": predicted_net_edge,
        "final_entry_edge_bps": final_entry_edge,
        "signal_component_bps": signal_component,
        "market_regime_component_bps": market_regime_component,
        "execution_component_bps": execution_component,
        "sizing_component_bps": sizing_component,
        "time_decay_component_bps": time_decay_component,
        "unexplained_component_bps": unexplained,
        "outcome_label": label,
        "raw_json": trade,
    }


def record_outcome_attribution(
    trade: dict[str, Any],
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.outcome_attribution_enabled):
        return {"status": "disabled"}

    event = outcome_attribution_from_trade(trade)
    if event is None:
        return {"status": "skipped", "message": "Insufficient trade outcome data"}

    initialize_outcome_attribution_db(db_path)
    path = _db_path(db_path)

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO outcome_attribution_events (
                created_at, symbol, execution_mode, entry_time, exit_time,
                entry_price, exit_price, quantity, gross_return_bps,
                realized_return_bps, realized_risk_cost_bps, trading_cost_bps,
                slippage_cost_bps, realized_net_edge_bps,
                predicted_net_edge_bps, final_entry_edge_bps,
                signal_component_bps, market_regime_component_bps,
                execution_component_bps, sizing_component_bps,
                time_decay_component_bps, unexplained_component_bps,
                outcome_label, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["created_at"],
                event["symbol"],
                event["execution_mode"],
                event["entry_time"],
                event["exit_time"],
                event["entry_price"],
                event["exit_price"],
                event["quantity"],
                event["gross_return_bps"],
                event["realized_return_bps"],
                event["realized_risk_cost_bps"],
                event["trading_cost_bps"],
                event["slippage_cost_bps"],
                event["realized_net_edge_bps"],
                event["predicted_net_edge_bps"],
                event["final_entry_edge_bps"],
                event["signal_component_bps"],
                event["market_regime_component_bps"],
                event["execution_component_bps"],
                event["sizing_component_bps"],
                event["time_decay_component_bps"],
                event["unexplained_component_bps"],
                event["outcome_label"],
                json.dumps(event["raw_json"], ensure_ascii=False, default=str),
            ),
        )
        _prune_old_records(conn)
        conn.commit()

    return {
        "status": "recorded",
        "symbol": event["symbol"],
        "outcome_label": event["outcome_label"],
        "realized_net_edge_bps": round(event["realized_net_edge_bps"], 4),
        "unexplained_component_bps": round(event["unexplained_component_bps"], 4),
    }


def _prune_old_records(conn: sqlite3.Connection) -> None:
    max_records = int(settings.outcome_attribution_max_records or 5000)
    if max_records <= 0:
        return

    conn.execute(
        """
        DELETE FROM outcome_attribution_events
        WHERE id NOT IN (
            SELECT id
            FROM outcome_attribution_events
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        )
        """,
        (max_records,),
    )


def outcome_attribution_summary(
    *,
    limit: int | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    if not bool(settings.outcome_attribution_enabled):
        return {"status": "disabled"}

    path = _db_path(db_path)
    if not path.exists():
        return {"status": "empty"}

    limit = max(
        10,
        min(int(limit or settings.outcome_attribution_recent_limit or 200), 1000),
    )

    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM outcome_attribution_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as exc:
        return {"status": "error", "message": str(exc)}

    if not rows:
        return {"status": "empty"}

    events = [dict(row) for row in rows]

    def avg(key: str) -> float:
        values = [
            float(item[key])
            for item in events
            if item.get(key) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    wins = sum(1 for item in events if item.get("outcome_label") == "win")
    losses = sum(1 for item in events if item.get("outcome_label") == "loss")

    return {
        "status": "ready",
        "sample_count": len(events),
        "win_count": wins,
        "loss_count": losses,
        "win_rate": round(wins / len(events), 6) if events else None,
        "avg_realized_net_edge_bps": round(avg("realized_net_edge_bps"), 4),
        "avg_signal_component_bps": round(avg("signal_component_bps"), 4),
        "avg_market_regime_component_bps": round(avg("market_regime_component_bps"), 4),
        "avg_execution_component_bps": round(avg("execution_component_bps"), 4),
        "avg_sizing_component_bps": round(avg("sizing_component_bps"), 4),
        "avg_time_decay_component_bps": round(avg("time_decay_component_bps"), 4),
        "avg_unexplained_component_bps": round(avg("unexplained_component_bps"), 4),
    }
