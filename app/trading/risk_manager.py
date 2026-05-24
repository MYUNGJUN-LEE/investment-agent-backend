from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.config import settings
from app.models import PaperRunRequest
from app.trading import cost_model, performance


@dataclass(frozen=True)
class RiskLimits:
    max_order_amount: float = 1_000_000
    portfolio_value: float = 10_000_000
    max_symbol_weight: float = 0.25
    max_sector_weight: float = 0.4
    max_trade_loss_pct: float = 0.01
    max_daily_loss_amount: float = 300_000
    max_weekly_loss_pct: float = 0.06
    max_monthly_loss_pct: float = 0.10
    max_consecutive_losses: int = 3
    new_entry_cutoff_time: time = time(15, 10)
    max_leverage: float = 1.0
    max_orders_per_minute: int = 5
    max_orders_per_day: int = 50
    duplicate_order_window_seconds: int = 60
    min_reward_risk_ratio: float = 1.5
    min_sharpe_ratio: float = 1.0
    preferred_sharpe_ratio: float = 1.5
    max_mdd: float = 0.2
    min_profit_factor: float = 1.2


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    code: str
    message: str
    checks: dict[str, Any]


DEFAULT_LIMITS = RiskLimits()


def recommend_order_quantity(
    *,
    price: float,
    stop_loss: float,
    account_equity: float,
    risk_per_trade: float,
    cash_available: float | None = None,
    limits: RiskLimits | None = None,
) -> dict[str, Any]:
    """Recommend an entry quantity from account risk and stop distance."""
    limits = limits or DEFAULT_LIMITS
    risk_per_share = abs(float(price) - float(stop_loss))
    risk_amount = float(account_equity) * float(risk_per_trade)
    risk_quantity = int(risk_amount / risk_per_share) if risk_per_share else 0
    cash_quantity = (
        int(float(cash_available) / float(price))
        if cash_available is not None and price > 0
        else None
    )
    max_order_quantity = int(limits.max_order_amount / float(price)) if price > 0 else 0
    caps = [risk_quantity, max_order_quantity]
    if cash_quantity is not None:
        caps.append(cash_quantity)
    recommended_quantity = max(0, min(caps)) if caps else 0
    return {
        "recommended_quantity": recommended_quantity,
        "risk_quantity": risk_quantity,
        "cash_quantity": cash_quantity,
        "max_order_quantity": max_order_quantity,
        "account_equity": float(account_equity),
        "risk_per_trade": float(risk_per_trade),
        "risk_amount": round(risk_amount, 2),
        "entry_price": float(price),
        "stop_loss": float(stop_loss),
        "risk_per_share": round(risk_per_share, 6),
        "estimated_amount": round(recommended_quantity * float(price), 2),
        "formula": "account_equity * risk_per_trade / abs(entry_price - stop_price)",
    }


def approve_order(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    side: str,
    quantity: int,
    now: datetime | str | None = None,
    limits: RiskLimits | None = None,
) -> RiskDecision:
    """Approve or reject an order before paper execution."""
    limits, dynamic_limits = _dynamic_risk_limits(
        conn=conn,
        req=req,
        limits=limits or DEFAULT_LIMITS,
    )
    side = side.upper()
    now_dt = _coerce_datetime(now)
    amount = round(req.price * quantity, 2)
    daily_loss = _daily_realized_loss(conn, now_dt.date())
    weekly_loss = _realized_loss_since(conn, now_dt - timedelta(days=7))
    monthly_loss = _realized_loss_since(conn, now_dt - timedelta(days=30))
    consecutive_losses = _consecutive_stop_losses(conn)
    symbol_exposure_limit = limits.portfolio_value * limits.max_symbol_weight
    sector_exposure_limit = limits.portfolio_value * limits.max_sector_weight
    projected_symbol_exposure = _projected_symbol_exposure(conn, req, side, quantity)
    projected_sector_exposure = _projected_sector_exposure(conn, req, side, quantity)
    projected_daily_loss = daily_loss + _projected_order_loss(conn, req, side, quantity)
    projected_trade_loss = _projected_trade_loss(req, quantity, limits)
    max_trade_loss_amount = _account_equity(req, limits) * limits.max_trade_loss_pct
    position_sizing = _position_sizing(req, quantity, limits)
    minute_order_count = _order_count_since(conn, now_dt - timedelta(minutes=1))
    daily_order_count = _order_count_since(conn, datetime.combine(now_dt.date(), time.min))
    edge_decision = (
        cost_model.evaluate_entry_edge(req=req, quantity=quantity)
        if side == "BUY" and quantity > 0
        else None
    )
    performance_metrics = performance.calculate_performance_metrics(
        conn=conn,
        symbol=None,
        market=req.market,
        strategy_type=req.strategy_type,
        portfolio_value=limits.portfolio_value,
        beta=req.market_beta,
    )

    checks = {
        "amount": amount,
        "max_order_amount": limits.max_order_amount,
        "daily_loss": daily_loss,
        "projected_daily_loss": projected_daily_loss,
        "max_daily_loss_amount": limits.max_daily_loss_amount,
        "weekly_loss": weekly_loss,
        "monthly_loss": monthly_loss,
        "max_weekly_loss_amount": _account_equity(req, limits) * limits.max_weekly_loss_pct,
        "max_monthly_loss_amount": _account_equity(req, limits) * limits.max_monthly_loss_pct,
        "projected_trade_loss": projected_trade_loss,
        "max_trade_loss_amount": max_trade_loss_amount,
        "consecutive_losses": consecutive_losses,
        "max_consecutive_losses": limits.max_consecutive_losses,
        "projected_symbol_exposure": projected_symbol_exposure,
        "symbol_exposure_limit": symbol_exposure_limit,
        "projected_sector_exposure": projected_sector_exposure,
        "sector_exposure_limit": sector_exposure_limit,
        "entry_cutoff_time": limits.new_entry_cutoff_time.isoformat(timespec="minutes"),
        "minute_order_count": minute_order_count,
        "max_orders_per_minute": limits.max_orders_per_minute,
        "daily_order_count": daily_order_count,
        "max_orders_per_day": limits.max_orders_per_day,
        "position_sizing": position_sizing,
        "edge": edge_decision,
        "performance_metrics": performance_metrics,
        "dynamic_limits": dynamic_limits,
    }

    if _emergency_stop_active():
        return _reject("emergency_stop_active", "Emergency stop is active", checks)

    if quantity <= 0:
        return _reject("invalid_quantity", "Order quantity must be positive", checks)

    if minute_order_count >= limits.max_orders_per_minute:
        return _reject("minute_order_limit_exceeded", "Minute order limit exceeded", checks)

    if daily_order_count >= limits.max_orders_per_day:
        return _reject("daily_order_limit_exceeded", "Daily order limit exceeded", checks)

    if req.leverage > limits.max_leverage:
        return _reject("leverage_limit_exceeded", "Leverage is above the allowed limit", checks)

    abnormal_price = _abnormal_price(req)
    if abnormal_price:
        return _reject(abnormal_price["code"], abnormal_price["message"], checks)

    if amount > limits.max_order_amount:
        return _reject(
            "max_order_amount_exceeded",
            "Order amount exceeds the per-order limit",
            checks,
        )

    if projected_daily_loss > limits.max_daily_loss_amount:
        return _reject(
            "daily_loss_limit_exceeded",
            "Projected daily realized loss exceeds the daily loss limit",
            checks,
        )

    if weekly_loss > checks["max_weekly_loss_amount"]:
        return _reject("weekly_loss_limit_exceeded", "Weekly loss limit exceeded", checks)

    if monthly_loss > checks["max_monthly_loss_amount"]:
        return _reject("monthly_loss_limit_exceeded", "Monthly loss limit exceeded", checks)

    if projected_trade_loss > max_trade_loss_amount:
        return _reject(
            "trade_loss_limit_exceeded",
            "Projected single-trade loss exceeds the per-trade loss limit",
            checks,
        )

    if side == "BUY":
        duplicate = _duplicate_order_exists(conn, req, side, now_dt, limits)
        if duplicate:
            return _reject("duplicate_order_detected", "Duplicate order detected", checks)

        if req.cash_available is not None and amount > req.cash_available:
            return _reject("cash_available_exceeded", "Order exceeds available cash", checks)

        if now_dt.time() >= limits.new_entry_cutoff_time:
            return _reject(
                "entry_cutoff_time_reached",
                "New entries are blocked before market close",
                checks,
            )

        if consecutive_losses >= limits.max_consecutive_losses:
            return _reject(
                "consecutive_stop_loss_limit_reached",
                "New entries are blocked after consecutive stop losses",
                checks,
            )

        if projected_symbol_exposure > symbol_exposure_limit:
            return _reject(
                "symbol_weight_limit_exceeded",
                "Projected symbol exposure exceeds the per-symbol weight limit",
                checks,
            )

        if req.sector and projected_sector_exposure > sector_exposure_limit:
            return _reject(
                "sector_weight_limit_exceeded",
                "Projected sector exposure exceeds the per-sector weight limit",
                checks,
            )

        if position_sizing and not position_sizing["approved"]:
            return _reject(
                "position_size_exceeded",
                position_sizing["message"],
                checks,
            )

        if edge_decision and not edge_decision["approved"]:
            return _reject(
                edge_decision["code"],
                edge_decision["message"],
                checks,
            )

        performance_rejection = _performance_rejection(performance_metrics, limits)
        if performance_rejection:
            return _reject(
                performance_rejection["code"],
                performance_rejection["message"],
                checks,
            )

    return RiskDecision(
        approved=True,
        code="approved",
        message="Risk checks passed",
        checks=checks,
    )


def _dynamic_risk_limits(
    *,
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    limits: RiskLimits,
) -> tuple[RiskLimits, dict[str, Any]]:
    if not settings.dynamic_risk_limits_enabled:
        return limits, {"enabled": False, "multiplier": 1.0}

    multiplier = 1.0
    reasons: list[str] = []
    market_regime = _market_regime(req)
    if market_regime in {"bear", "risk_off", "downtrend", "weak"}:
        factor = float(settings.dynamic_risk_bear_multiplier or 0.6)
        multiplier *= factor
        reasons.append(f"market_regime={market_regime} factor={factor:.2f}")
    elif market_regime in {"bull", "risk_on", "uptrend", "strong"}:
        factor = float(settings.dynamic_risk_bull_multiplier or 1.05)
        multiplier *= factor
        reasons.append(f"market_regime={market_regime} factor={factor:.2f}")
    elif market_regime in {"unknown", ""}:
        factor = float(settings.dynamic_risk_unknown_multiplier or 1.0)
        multiplier *= factor
        reasons.append(f"market_regime=unknown factor={factor:.2f}")

    atr_pct = _latest_atr_pct(req)
    high_atr = max(0.001, float(settings.dynamic_risk_high_atr_pct or 0.06))
    low_atr = max(0.001, float(settings.dynamic_risk_low_atr_pct or 0.025))
    if atr_pct is not None:
        if atr_pct >= high_atr:
            factor = max(0.45, high_atr / atr_pct)
            multiplier *= factor
            reasons.append(f"atr_pct={atr_pct:.4f} high factor={factor:.2f}")
        elif atr_pct <= low_atr:
            multiplier *= 1.05
            reasons.append(f"atr_pct={atr_pct:.4f} low factor=1.05")

    max_corr = _max_portfolio_correlation(conn, req.symbol)
    high_corr = float(settings.dynamic_risk_high_correlation or 0.75)
    if max_corr is not None and max_corr >= high_corr:
        factor = max(0.55, 1.0 - (max_corr - high_corr))
        multiplier *= factor
        reasons.append(f"portfolio_correlation={max_corr:.3f} factor={factor:.2f}")

    min_multiplier = float(settings.dynamic_risk_min_multiplier or 0.35)
    max_multiplier = float(settings.dynamic_risk_max_multiplier or 1.15)
    multiplier = max(min_multiplier, min(max_multiplier, multiplier))
    adjusted = replace(
        limits,
        max_order_amount=limits.max_order_amount * multiplier,
        max_trade_loss_pct=limits.max_trade_loss_pct * multiplier,
        max_daily_loss_amount=limits.max_daily_loss_amount * multiplier,
        max_symbol_weight=limits.max_symbol_weight * multiplier,
        max_sector_weight=limits.max_sector_weight * multiplier,
    )
    return adjusted, {
        "enabled": True,
        "multiplier": round(multiplier, 6),
        "base": {
            "max_order_amount": limits.max_order_amount,
            "max_trade_loss_pct": limits.max_trade_loss_pct,
            "max_daily_loss_amount": limits.max_daily_loss_amount,
            "max_symbol_weight": limits.max_symbol_weight,
            "max_sector_weight": limits.max_sector_weight,
        },
        "adjusted": {
            "max_order_amount": round(adjusted.max_order_amount, 2),
            "max_trade_loss_pct": round(adjusted.max_trade_loss_pct, 6),
            "max_daily_loss_amount": round(adjusted.max_daily_loss_amount, 2),
            "max_symbol_weight": round(adjusted.max_symbol_weight, 6),
            "max_sector_weight": round(adjusted.max_sector_weight, 6),
        },
        "market_regime": market_regime,
        "atr_pct": atr_pct,
        "max_portfolio_correlation": max_corr,
        "reasons": reasons,
    }


def _market_regime(req: PaperRunRequest) -> str:
    if req.market_regime:
        return str(req.market_regime).lower()
    try:
        from app.storage.market_data import get_latest_market_context

        context = get_latest_market_context() or {}
    except Exception:
        context = {}
    return str(context.get("market_regime") or "unknown").lower()


def _latest_atr_pct(req: PaperRunRequest) -> float | None:
    if req.expected_loss_bps is not None:
        value = _to_float(req.expected_loss_bps)
        if value is not None and value > 0:
            return value / 10_000
    path = Path(settings.universe_scanner_db_path)
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT raw_json
                FROM universe_price_snapshots
                WHERE symbol = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (req.symbol,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    raw = _parse_json(row["raw_json"], {})
    latest = raw.get("latest_technical_features") or {}
    atr_pct = _to_float(latest.get("atr_14_pct"))
    if atr_pct is not None:
        return atr_pct
    features = raw.get("technical_features") or []
    if features and isinstance(features[-1], dict):
        return _to_float(features[-1].get("atr_14_pct"))
    return None


def _max_portfolio_correlation(
    conn: sqlite3.Connection,
    symbol: str,
) -> float | None:
    try:
        rows = conn.execute(
            """
            SELECT symbol
            FROM positions
            WHERE quantity > 0
              AND symbol != ?
            LIMIT 10
            """,
            (symbol,),
        ).fetchall()
    except sqlite3.Error:
        return None
    held_symbols = []
    for row in rows:
        value = row["symbol"] if hasattr(row, "keys") else row[0]
        if value:
            held_symbols.append(str(value))
    if not held_symbols:
        return None
    target_returns = _recent_snapshot_returns(symbol)
    if len(target_returns) < 5:
        return None
    correlations = []
    for held_symbol in held_symbols:
        held_returns = _recent_snapshot_returns(held_symbol)
        correlation = _pearson_tail(target_returns, held_returns)
        if correlation is not None:
            correlations.append(correlation)
    return max(correlations) if correlations else None


def _recent_snapshot_returns(symbol: str, limit: int = 80) -> list[float]:
    path = Path(settings.universe_scanner_db_path)
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT current_price
                FROM universe_price_snapshots
                WHERE symbol = ?
                  AND current_price IS NOT NULL
                  AND current_price > 0
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (symbol, max(2, int(limit))),
            ).fetchall()
    except sqlite3.Error:
        return []
    prices = [float(row["current_price"]) for row in reversed(rows)]
    returns = []
    for previous, current in zip(prices, prices[1:]):
        if previous > 0:
            returns.append((current - previous) / previous)
    return returns


def _pearson_tail(left: list[float], right: list[float]) -> float | None:
    size = min(len(left), len(right))
    if size < 5:
        return None
    x_values = left[-size:]
    y_values = right[-size:]
    x_mean = sum(x_values) / size
    y_mean = sum(y_values) / size
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = sum((x - x_mean) ** 2 for x in x_values) ** 0.5
    y_denominator = sum((y - y_mean) ** 2 for y in y_values) ** 0.5
    denominator = x_denominator * y_denominator
    if denominator <= 0:
        return None
    return max(-1.0, min(1.0, numerator / denominator))


def _daily_realized_loss(conn: sqlite3.Connection, day: date) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(
            CASE
              WHEN (CASE WHEN net_realized_pnl != 0 THEN net_realized_pnl ELSE realized_pnl END) < 0
              THEN -(CASE WHEN net_realized_pnl != 0 THEN net_realized_pnl ELSE realized_pnl END)
              ELSE 0
            END
        ), 0)
        FROM paper_orders
        WHERE status IN ('FILLED', 'PARTIALLY_FILLED')
          AND side = 'SELL'
          AND substr(created_at, 1, 10) = ?
        """,
        (day.isoformat(),),
    ).fetchone()
    return float(row[0] or 0)


def _realized_loss_since(conn: sqlite3.Connection, start: datetime) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(
            CASE
              WHEN (CASE WHEN net_realized_pnl != 0 THEN net_realized_pnl ELSE realized_pnl END) < 0
              THEN -(CASE WHEN net_realized_pnl != 0 THEN net_realized_pnl ELSE realized_pnl END)
              ELSE 0
            END
        ), 0)
        FROM paper_orders
        WHERE status IN ('FILLED', 'PARTIALLY_FILLED')
          AND side = 'SELL'
          AND created_at >= ?
        """,
        (start.isoformat(timespec="seconds"),),
    ).fetchone()
    return float(row[0] or 0)


def _consecutive_stop_losses(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT CASE WHEN net_realized_pnl != 0 THEN net_realized_pnl ELSE realized_pnl END AS pnl
        FROM paper_orders
        WHERE status IN ('FILLED', 'PARTIALLY_FILLED')
          AND side = 'SELL'
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()

    count = 0
    for row in rows:
        if float(row["pnl"]) < 0:
            count += 1
        else:
            break
    return count


def _projected_symbol_exposure(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    side: str,
    quantity: int,
) -> float:
    row = conn.execute(
        "SELECT quantity FROM positions WHERE symbol = ?",
        (req.symbol,),
    ).fetchone()
    current_qty = int(row["quantity"]) if row else 0
    if side == "BUY":
        projected_qty = current_qty + quantity
    else:
        projected_qty = max(0, current_qty - quantity)
    return round(projected_qty * req.price, 2)


def _projected_sector_exposure(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    side: str,
    quantity: int,
) -> float:
    if not req.sector:
        return 0.0
    row = conn.execute(
        """
        SELECT COALESCE(SUM(quantity * avg_price), 0)
        FROM positions
        WHERE sector = ? AND symbol != ?
        """,
        (req.sector, req.symbol),
    ).fetchone()
    other_sector_exposure = float(row[0] or 0)
    return round(
        other_sector_exposure + _projected_symbol_exposure(conn, req, side, quantity),
        2,
    )


def _projected_order_loss(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    side: str,
    quantity: int,
) -> float:
    if side != "SELL":
        return 0.0

    row = conn.execute(
        "SELECT quantity, avg_price, cost_basis FROM positions WHERE symbol = ?",
        (req.symbol,),
    ).fetchone()
    if not row:
        return 0.0

    current_qty = int(row["quantity"] or 0)
    if current_qty <= 0:
        return 0.0

    cost_basis = float(row["cost_basis"] or 0)
    if cost_basis <= 0:
        cost_basis = float(row["avg_price"]) * current_qty
    allocated_cost_basis = cost_basis * min(quantity, current_qty) / current_qty
    sell_cost = cost_model.estimate_order_cost(req=req, side="SELL", quantity=quantity)
    net_proceeds = sell_cost.requested_amount - sell_cost.total_cost
    realized_pnl = net_proceeds - allocated_cost_basis
    return abs(realized_pnl) if realized_pnl < 0 else 0.0


def _projected_trade_loss(
    req: PaperRunRequest,
    quantity: int,
    limits: RiskLimits,
) -> float:
    if req.stop_loss is None:
        return 0.0
    per_share_loss = abs(req.price - req.stop_loss)
    return round(per_share_loss * quantity, 2)


def _position_sizing(
    req: PaperRunRequest,
    quantity: int,
    limits: RiskLimits,
) -> dict[str, Any] | None:
    if req.stop_loss is None:
        return None
    risk_per_trade = req.risk_per_trade or _default_risk_per_trade(req.risk_level)
    account_equity = _account_equity(req, limits)
    risk_amount = account_equity * risk_per_trade
    risk_per_share = abs(req.price - req.stop_loss)
    max_quantity = int(risk_amount / risk_per_share) if risk_per_share else 0
    approved = quantity <= max_quantity
    return {
        "approved": approved,
        "account_equity": account_equity,
        "risk_per_trade": risk_per_trade,
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 6),
        "max_quantity": max_quantity,
        "requested_quantity": quantity,
        "formula": "account_equity * risk_per_trade / abs(entry_price - stop_price)",
        "message": "Position size is within risk budget"
        if approved
        else f"Requested quantity {quantity} exceeds position sizing max {max_quantity}",
    }


def _default_risk_per_trade(risk_level: str) -> float:
    return {"low": 0.002, "medium": 0.005, "high": 0.01}.get(risk_level, 0.005)


def _account_equity(req: PaperRunRequest, limits: RiskLimits) -> float:
    return float(req.account_equity or limits.portfolio_value)


def _order_count_since(conn: sqlite3.Connection, start: datetime) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM paper_orders
        WHERE created_at >= ?
        """,
        (start.isoformat(timespec="seconds"),),
    ).fetchone()
    return int(row[0] or 0)


def _duplicate_order_exists(
    conn: sqlite3.Connection,
    req: PaperRunRequest,
    side: str,
    now_dt: datetime,
    limits: RiskLimits,
) -> bool:
    since = now_dt - timedelta(seconds=limits.duplicate_order_window_seconds)
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM paper_orders
        WHERE symbol = ?
          AND side = ?
          AND status IN ('FILLED', 'PARTIALLY_FILLED', 'PENDING')
          AND created_at >= ?
        """,
        (req.symbol, side, since.isoformat(timespec="seconds")),
    ).fetchone()
    return int(row[0] or 0) > 0


def _abnormal_price(req: PaperRunRequest) -> dict[str, str] | None:
    reference_price = req.decision_price
    if reference_price is None:
        return None
    deviation_bps = abs(req.price - reference_price) / reference_price * 10_000
    if deviation_bps > settings.max_order_price_deviation_bps:
        return {
            "code": "abnormal_price_deviation",
            "message": (
                f"Order price deviation {deviation_bps:.1f}bps exceeds "
                f"{settings.max_order_price_deviation_bps:.1f}bps"
            ),
        }
    return None


def _emergency_stop_active() -> bool:
    return bool(settings.emergency_stop) or Path(settings.emergency_stop_file).exists()


def _performance_rejection(
    metrics: dict[str, Any],
    limits: RiskLimits,
) -> dict[str, str] | None:
    min_sample = settings.performance_min_trade_count
    if int(metrics.get("trade_count") or 0) < min_sample:
        return None

    sharpe = metrics.get("sharpe_ratio")
    if isinstance(sharpe, (int, float)) and sharpe < limits.min_sharpe_ratio:
        return {
            "code": "historical_sharpe_too_low",
            "message": f"Historical Sharpe {sharpe:.2f} is below {limits.min_sharpe_ratio:.2f}",
        }

    profit_factor = metrics.get("profit_factor")
    if (
        isinstance(profit_factor, (int, float))
        and profit_factor < limits.min_profit_factor
    ):
        return {
            "code": "historical_profit_factor_too_low",
            "message": (
                f"Historical profit factor {profit_factor:.2f} is below "
                f"{limits.min_profit_factor:.2f}"
            ),
        }

    mdd = metrics.get("mdd")
    if isinstance(mdd, (int, float)) and abs(mdd) > limits.max_mdd:
        return {
            "code": "historical_mdd_too_high",
            "message": f"Historical MDD {mdd:.2%} exceeds {limits.max_mdd:.2%}",
        }

    return None


def _coerce_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now()


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _reject(code: str, message: str, checks: dict[str, Any]) -> RiskDecision:
    return RiskDecision(
        approved=False,
        code=code,
        message=message,
        checks=checks,
    )
