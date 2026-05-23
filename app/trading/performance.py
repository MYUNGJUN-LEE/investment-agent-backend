from __future__ import annotations

from datetime import datetime
import json
import math
import sqlite3
from typing import Any


PERFORMANCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scope TEXT NOT NULL,
    symbol TEXT,
    market TEXT,
    strategy_type TEXT,
    trade_count INTEGER NOT NULL,
    cagr REAL,
    mdd REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    profit_factor REAL,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    avg_win_loss_ratio REAL,
    expectancy REAL,
    turnover REAL,
    exposure REAL,
    beta REAL,
    tail_loss REAL,
    raw_json TEXT NOT NULL
);
"""


def initialize_performance_db(conn: sqlite3.Connection) -> None:
    conn.executescript(PERFORMANCE_SCHEMA_SQL)


def record_performance_snapshot(
    conn: sqlite3.Connection,
    created_at: str,
    symbol: str | None,
    market: str | None,
    strategy_type: str | None,
    portfolio_value: float,
    beta: float | None = None,
) -> dict[str, Any]:
    metrics = calculate_performance_metrics(
        conn=conn,
        symbol=symbol,
        market=market,
        strategy_type=strategy_type,
        portfolio_value=portfolio_value,
        beta=beta,
    )
    conn.execute(
        """
        INSERT INTO performance_snapshots (
            created_at, scope, symbol, market, strategy_type, trade_count,
            cagr, mdd, sharpe_ratio, sortino_ratio, calmar_ratio,
            profit_factor, win_rate, avg_win, avg_loss, avg_win_loss_ratio,
            expectancy, turnover, exposure, beta, tail_loss, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            "symbol_strategy",
            symbol,
            market,
            strategy_type,
            metrics["trade_count"],
            metrics["cagr"],
            metrics["mdd"],
            metrics["sharpe_ratio"],
            metrics["sortino_ratio"],
            metrics["calmar_ratio"],
            metrics["profit_factor"],
            metrics["win_rate"],
            metrics["avg_win"],
            metrics["avg_loss"],
            metrics["avg_win_loss_ratio"],
            metrics["expectancy"],
            metrics["turnover"],
            metrics["exposure"],
            metrics["beta"],
            metrics["tail_loss"],
            json.dumps(metrics, ensure_ascii=False, default=str),
        ),
    )
    return metrics


def calculate_performance_metrics(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    market: str | None = None,
    strategy_type: str | None = None,
    portfolio_value: float = 1.0,
    beta: float | None = None,
) -> dict[str, Any]:
    orders = _filled_orders(conn, symbol=symbol)
    sells = [order for order in orders if order["side"] == "SELL"]
    trade_pnls = [_net_pnl(order) for order in sells]
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    trade_count = len(trade_pnls)
    profit_factor = gross_profit / gross_loss if gross_loss else (None if not gross_profit else None)
    win_rate = len(wins) / trade_count if trade_count else None
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    avg_win_loss_ratio = (
        avg_win / abs(avg_loss)
        if avg_win is not None and avg_loss not in (None, 0)
        else None
    )
    expectancy = sum(trade_pnls) / trade_count if trade_count else None
    turnover = (
        sum(float(order["amount"] or 0) for order in orders) / portfolio_value
        if portfolio_value
        else None
    )
    exposure = _current_exposure(conn, portfolio_value, symbol=symbol)
    daily_pnls = _daily_pnls(sells)
    daily_returns = [pnl / portfolio_value for pnl in daily_pnls.values()] if portfolio_value else []
    sharpe = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)
    tail_loss = min(daily_pnls.values()) if daily_pnls else None
    cagr = _cagr(orders, sum(trade_pnls), portfolio_value)
    mdd = _max_drawdown(daily_pnls, portfolio_value)
    calmar = cagr / abs(mdd) if cagr is not None and mdd not in (None, 0) else None

    return {
        "symbol": symbol,
        "market": market,
        "strategy_type": strategy_type,
        "trade_count": trade_count,
        "cagr": _round_or_none(cagr),
        "mdd": _round_or_none(mdd),
        "sharpe_ratio": _round_or_none(sharpe),
        "sortino_ratio": _round_or_none(sortino),
        "calmar_ratio": _round_or_none(calmar),
        "profit_factor": _round_or_none(profit_factor),
        "win_rate": _round_or_none(win_rate),
        "avg_win": _round_or_none(avg_win),
        "avg_loss": _round_or_none(avg_loss),
        "avg_win_loss_ratio": _round_or_none(avg_win_loss_ratio),
        "expectancy": _round_or_none(expectancy),
        "turnover": _round_or_none(turnover),
        "exposure": _round_or_none(exposure),
        "beta": beta,
        "tail_loss": _round_or_none(tail_loss),
    }


def _filled_orders(
    conn: sqlite3.Connection,
    symbol: str | None = None,
) -> list[sqlite3.Row]:
    query = """
        SELECT *
        FROM paper_orders
        WHERE status IN ('FILLED', 'PARTIALLY_FILLED')
    """
    params: tuple[Any, ...] = ()
    if symbol:
        query += " AND symbol = ?"
        params = (symbol,)
    query += " ORDER BY created_at, id"
    return conn.execute(query, params).fetchall()


def _net_pnl(order: sqlite3.Row) -> float:
    keys = set(order.keys())
    if (
        "net_realized_pnl" in keys
        and order["net_realized_pnl"] is not None
        and float(order["net_realized_pnl"]) != 0
    ):
        return float(order["net_realized_pnl"])
    return float(order["realized_pnl"] or 0)


def _daily_pnls(orders: list[sqlite3.Row]) -> dict[str, float]:
    daily: dict[str, float] = {}
    for order in orders:
        day = str(order["created_at"])[:10]
        daily[day] = daily.get(day, 0.0) + _net_pnl(order)
    return daily


def _current_exposure(
    conn: sqlite3.Connection,
    portfolio_value: float,
    symbol: str | None = None,
) -> float | None:
    query = "SELECT quantity, avg_price, cost_basis FROM positions WHERE quantity > 0"
    params: tuple[Any, ...] = ()
    if symbol:
        query += " AND symbol = ?"
        params = (symbol,)
    rows = conn.execute(query, params).fetchall()
    exposure_amount = 0.0
    for row in rows:
        cost_basis = float(row["cost_basis"] or 0)
        if cost_basis <= 0:
            cost_basis = float(row["quantity"] or 0) * float(row["avg_price"] or 0)
        exposure_amount += cost_basis
    return exposure_amount / portfolio_value if portfolio_value else None


def _sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return mean_return / std * math.sqrt(252)


def _sortino(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    downside = [value for value in returns if value < 0]
    if not downside:
        return None
    mean_return = sum(returns) / len(returns)
    downside_std = math.sqrt(sum(value**2 for value in downside) / len(downside))
    if downside_std == 0:
        return None
    return mean_return / downside_std * math.sqrt(252)


def _cagr(
    orders: list[sqlite3.Row],
    total_pnl: float,
    portfolio_value: float,
) -> float | None:
    if len(orders) < 2 or portfolio_value <= 0:
        return None
    start = _parse_dt(orders[0]["created_at"])
    end = _parse_dt(orders[-1]["created_at"])
    days = max((end - start).days, 1)
    years = days / 365
    ending_value = portfolio_value + total_pnl
    if ending_value <= 0:
        return -1.0
    return (ending_value / portfolio_value) ** (1 / years) - 1


def _max_drawdown(
    daily_pnls: dict[str, float],
    portfolio_value: float,
) -> float | None:
    if not daily_pnls or portfolio_value <= 0:
        return None
    equity = portfolio_value
    peak = equity
    max_drawdown = 0.0
    for day in sorted(daily_pnls):
        equity += daily_pnls[day]
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak else 0
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
