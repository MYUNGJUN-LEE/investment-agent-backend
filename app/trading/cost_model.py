from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.models import PaperRunRequest


BPS = 10_000


@dataclass(frozen=True)
class OrderCost:
    requested_price: float
    effective_price: float
    requested_amount: float
    effective_amount: float
    commission: float
    tax: float
    spread_cost: float
    slippage_cost: float
    fx_cost: float
    financing_cost: float
    borrow_cost: float
    total_cost: float
    total_cost_bps: float
    fill_probability: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: round(value, 6) for key, value in data.items()}


def estimate_order_cost(
    req: PaperRunRequest,
    side: str,
    quantity: int,
) -> OrderCost:
    side = side.upper()
    requested_price = float(req.price)
    requested_amount = requested_price * quantity
    spread_bps = _non_negative(req.spread_bps)
    slippage_bps = _non_negative(req.slippage_bps)
    adverse_bps = spread_bps + slippage_bps
    direction = 1 if side == "BUY" else -1
    effective_price = requested_price * (1 + direction * adverse_bps / BPS)
    effective_amount = effective_price * quantity
    commission_rate = _commission_rate(req)
    commission = effective_amount * commission_rate
    tax = effective_amount * _sell_tax_rate(req, side)
    fx_cost = requested_amount * _fx_spread_bps(req) / BPS
    financing_cost = _financing_cost(req, requested_amount)
    borrow_cost = _borrow_cost(req, requested_amount, side)
    spread_cost = requested_amount * spread_bps / BPS
    slippage_cost = requested_amount * slippage_bps / BPS
    total_cost = (
        commission
        + tax
        + spread_cost
        + slippage_cost
        + fx_cost
        + financing_cost
        + borrow_cost
    )

    return OrderCost(
        requested_price=round(requested_price, 6),
        effective_price=round(effective_price, 6),
        requested_amount=round(requested_amount, 6),
        effective_amount=round(effective_amount, 6),
        commission=round(commission, 6),
        tax=round(tax, 6),
        spread_cost=round(spread_cost, 6),
        slippage_cost=round(slippage_cost, 6),
        fx_cost=round(fx_cost, 6),
        financing_cost=round(financing_cost, 6),
        borrow_cost=round(borrow_cost, 6),
        total_cost=round(total_cost, 6),
        total_cost_bps=round(total_cost / requested_amount * BPS, 4)
        if requested_amount
        else 0,
        fill_probability=_fill_probability(req),
    )


def estimate_round_trip_cost(
    req: PaperRunRequest,
    quantity: int,
) -> dict[str, Any]:
    buy = estimate_order_cost(req, "BUY", quantity)
    sell = estimate_order_cost(req, "SELL", quantity)
    requested_amount = buy.requested_amount
    total_cost = buy.total_cost + sell.total_cost
    total_cost_bps = total_cost / requested_amount * BPS if requested_amount else 0
    return {
        "buy": buy.to_dict(),
        "sell": sell.to_dict(),
        "total_cost": round(total_cost, 6),
        "total_cost_bps": round(total_cost_bps, 4),
        "fill_probability": _fill_probability(req),
    }


def evaluate_entry_edge(
    req: PaperRunRequest,
    quantity: int,
) -> dict[str, Any]:
    round_trip = estimate_round_trip_cost(req, quantity)
    total_cost_bps = float(round_trip["total_cost_bps"])
    min_net_edge_bps = _min_net_edge_bps(req.strategy_type, req.risk_level)
    min_cost_multiple = _min_cost_multiple(req.risk_level)
    required_net_edge_bps = max(min_net_edge_bps, total_cost_bps * min_cost_multiple)
    gross_edge_bps, edge_source = _expected_gross_edge_bps(
        req=req,
        total_cost_bps=total_cost_bps,
        required_net_edge_bps=required_net_edge_bps,
    )
    fill_probability = _fill_probability(req)
    fill_adjusted_gross_edge_bps = gross_edge_bps * fill_probability
    net_edge_bps = fill_adjusted_gross_edge_bps - total_cost_bps
    reward_risk_ratio, reward_risk_source = _reward_risk_ratio(req)
    expected_sharpe = req.expected_sharpe

    blockers: list[str] = []
    if fill_probability < settings.min_fill_probability:
        blockers.append(
            f"Fill probability {fill_probability:.2f} is below {settings.min_fill_probability:.2f}"
        )
    if net_edge_bps < total_cost_bps * min_cost_multiple:
        blockers.append(
            f"Net edge {net_edge_bps:.2f}bps is below {min_cost_multiple}x total cost"
        )
    if net_edge_bps < min_net_edge_bps:
        blockers.append(
            f"Net edge {net_edge_bps:.2f}bps is below strategy minimum {min_net_edge_bps:.2f}bps"
        )
    if reward_risk_ratio < 1.5:
        blockers.append(
            f"Reward/risk {reward_risk_ratio:.2f} is below 1.5"
        )
    if expected_sharpe is not None and expected_sharpe < 1.0:
        blockers.append(f"Expected Sharpe {expected_sharpe:.2f} is below 1.0")

    return {
        "approved": not blockers,
        "code": "approved" if not blockers else "edge_requirement_not_met",
        "message": "Cost and edge checks passed"
        if not blockers
        else "; ".join(blockers),
        "blockers": blockers,
        "gross_edge_bps": round(gross_edge_bps, 4),
        "gross_edge_source": edge_source,
        "fill_adjusted_gross_edge_bps": round(fill_adjusted_gross_edge_bps, 4),
        "net_edge_bps": round(net_edge_bps, 4),
        "required_net_edge_bps": round(required_net_edge_bps, 4),
        "min_net_edge_bps": round(min_net_edge_bps, 4),
        "min_cost_multiple": min_cost_multiple,
        "reward_risk_ratio": round(reward_risk_ratio, 4),
        "reward_risk_source": reward_risk_source,
        "expected_sharpe": expected_sharpe,
        "round_trip_cost": round_trip,
    }


def _expected_gross_edge_bps(
    req: PaperRunRequest,
    total_cost_bps: float,
    required_net_edge_bps: float,
) -> tuple[float, str]:
    if req.expected_gross_edge_bps is not None:
        return max(0.0, float(req.expected_gross_edge_bps)), "request"

    confidence_adjustment = (float(req.confidence) - 0.5) * 100
    estimated = total_cost_bps + required_net_edge_bps + confidence_adjustment
    return max(0.0, estimated), "estimated_from_confidence"


def _reward_risk_ratio(req: PaperRunRequest) -> tuple[float, str]:
    if req.expected_win_bps is not None and req.expected_loss_bps is not None:
        loss = abs(float(req.expected_loss_bps))
        if loss == 0:
            return 0.0, "request"
        return max(0.0, float(req.expected_win_bps)) / loss, "request"
    return 1.6, "default_strategy_floor"


def _min_net_edge_bps(strategy_type: str, risk_level: str) -> float:
    if strategy_type == "daytrade":
        return {"low": 10.0, "medium": 7.5, "high": 5.0}.get(risk_level, 7.5)
    if strategy_type == "swing":
        return {"low": 100.0, "medium": 50.0, "high": 30.0}.get(risk_level, 50.0)
    return {"low": 120.0, "medium": 80.0, "high": 50.0}.get(risk_level, 80.0)


def _min_cost_multiple(risk_level: str) -> float:
    return {"low": 3.0, "medium": 2.5, "high": 2.0}.get(risk_level, 2.5)


def _commission_rate(req: PaperRunRequest) -> float:
    return _non_negative(req.commission_rate, settings.commission_rate)


def _sell_tax_rate(req: PaperRunRequest, side: str) -> float:
    if side != "SELL" or req.market != "KR":
        return 0.0
    return _non_negative(req.sell_tax_rate, settings.kr_stock_sell_tax_rate)


def _fx_spread_bps(req: PaperRunRequest) -> float:
    return 0.0


def _financing_cost(req: PaperRunRequest, amount: float) -> float:
    if req.leverage <= 1:
        return 0.0
    rate = _non_negative(req.margin_interest_rate, settings.margin_interest_annual_rate)
    borrowed_ratio = (req.leverage - 1) / req.leverage
    return amount * borrowed_ratio * rate * req.expected_holding_days / 365


def _borrow_cost(req: PaperRunRequest, amount: float, side: str) -> float:
    if side != "SELL" or not req.is_short:
        return 0.0
    rate = _non_negative(req.borrow_fee_rate, settings.short_borrow_annual_rate)
    return amount * rate * req.expected_holding_days / 365


def _fill_probability(req: PaperRunRequest) -> float:
    value = req.fill_probability
    if value is None:
        value = settings.default_fill_probability
    return max(0.0, min(1.0, float(value)))


def _non_negative(value: float | None, default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, float(value))
