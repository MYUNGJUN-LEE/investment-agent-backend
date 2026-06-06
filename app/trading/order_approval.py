from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import secrets
import sqlite3
from typing import Any

from app.config import settings
from app.models import (
    OrderConfirmRequest,
    OrderPreviewRequest,
    PaperRunRequest,
    PipelineRequest,
)
from app.services.pipeline import run_full_pipeline
from app.storage.sqlite import connect_sqlite, sqlite_write_with_retry
from app.strategies.rule_based import build_strategy_decision
from app.trading import paper_trading, risk_manager


class OrderApprovalError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_order_preview(
    req: OrderPreviewRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run analysis, build a strategy candidate, and store a confirmable preview."""
    req, quantity_recommendation = _resolve_preview_quantity(req)
    resolved_db_path = settings.storage_path(db_path or paper_trading.DEFAULT_DB_PATH)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_req = PipelineRequest(
        symbol=req.symbol,
        name=req.name,
        market=req.market,
        strategy_type=req.strategy_type,
        sector=req.sector,
        lookback_hours=req.lookback_hours,
        risk_level=req.risk_level,
    )
    pipeline_result = run_full_pipeline(pipeline_req)
    strategy_decision = build_strategy_decision(
        pipeline_result=pipeline_result,
        requested_action=req.requested_action,
        risk_level=req.risk_level,
    )
    strategy_decision["cost_inputs"] = _cost_inputs_from_preview(req, pipeline_result)
    strategy_decision["recommended_quantity"] = quantity_recommendation

    now = _now()
    amount = round(req.price * req.quantity, 2)
    risk_decision: dict[str, Any] | None = None
    status = "blocked"
    token = None
    message = "Strategy blocked this order preview"

    def operation() -> int:
        nonlocal risk_decision, status, token, message
        with connect_sqlite(resolved_db_path, row_factory=True) as conn:
            paper_trading.initialize_db(conn)

            if strategy_decision["approved"]:
                paper_req = _to_paper_request(req, strategy_decision)
                decision = risk_manager.approve_order(
                    conn=conn,
                    req=paper_req,
                    side=strategy_decision["side"],
                    quantity=req.quantity,
                    now=now,
                )
                risk_decision = _risk_decision_to_dict(decision)
                if decision.approved:
                    status = "pending"
                    token = secrets.token_urlsafe(24)
                    message = "Order preview is ready for user confirmation"
                else:
                    message = f"Risk blocked this order preview: {decision.message} ({decision.code})"
            else:
                risk_decision = None
                message = "; ".join(strategy_decision["blocking_reasons"])

            return _insert_preview(
                conn=conn,
                created_at=now,
                status=status,
                preview_token=token,
                req=req,
                strategy_decision=strategy_decision,
                risk_decision=risk_decision,
                pipeline_result=pipeline_result,
                message=message,
                amount=amount,
            )

    preview_id = sqlite_write_with_retry(operation)

    return {
        "status": status,
        "preview_id": preview_id,
        "preview_token": token,
        "symbol": req.symbol,
        "signal_type": strategy_decision["signal_type"],
        "side": strategy_decision["side"],
        "price": req.price,
        "quantity": req.quantity,
        "amount": amount,
        "recommended_quantity": quantity_recommendation,
        "message": message,
        "strategy_decision": strategy_decision,
        "risk_decision": risk_decision,
        "cost_edge_decision": (risk_decision or {}).get("checks", {}).get("edge")
        if risk_decision
        else None,
    }


def confirm_order_preview(
    req: OrderConfirmRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Confirm a pending preview and execute it through paper trading."""
    resolved_db_path = settings.storage_path(db_path or paper_trading.DEFAULT_DB_PATH)

    with connect_sqlite(resolved_db_path, row_factory=True) as conn:
        paper_trading.initialize_db(conn)
        preview = _get_preview(conn, req.preview_id)
        if not preview:
            raise OrderApprovalError("Order preview not found", status_code=404)
        if preview["status"] != "pending":
            raise OrderApprovalError("Order preview is not pending", status_code=409)
        if preview["preview_token"] != req.preview_token:
            raise OrderApprovalError("Invalid preview_token", status_code=403)

    strategy_decision = json.loads(preview["raw_strategy_decision"])
    cost_inputs = strategy_decision.get("cost_inputs") or {}
    paper_req = PaperRunRequest(
        symbol=preview["symbol"],
        name=preview["name"],
        market=preview["market"],
        strategy_type=preview["strategy_type"],
        risk_level=cost_inputs.get("risk_level", "medium"),
        signal_type=preview["signal_type"],
        price=float(preview["price"]),
        quantity=int(preview["quantity"]),
        confidence=float(preview["confidence"]),
        reason=preview["strategy_message"],
        source="order_preview",
        **_paper_cost_kwargs(cost_inputs),
    )
    paper_result = paper_trading.run_paper_once(paper_req, db_path=resolved_db_path)

    def operation() -> str:
        with connect_sqlite(resolved_db_path, row_factory=True) as conn:
            status = "confirmed" if paper_result["order_status"] == "FILLED" else "rejected"
            conn.execute(
                """
                UPDATE order_previews
                SET status = ?, confirmed_at = ?, paper_order_id = ?
                WHERE id = ?
                """,
                (
                    status,
                    _now(),
                    paper_result["order_id"],
                    req.preview_id,
                ),
            )
            return status

    status = sqlite_write_with_retry(operation)

    return {
        "status": status,
        "preview_id": req.preview_id,
        "execution_mode": req.execution_mode,
        "paper_result": paper_result,
    }


def _insert_preview(
    conn: sqlite3.Connection,
    created_at: str,
    status: str,
    preview_token: str | None,
    req: OrderPreviewRequest,
    strategy_decision: dict[str, Any],
    risk_decision: dict[str, Any] | None,
    pipeline_result: dict[str, Any],
    message: str,
    amount: float,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO order_previews (
            created_at, status, preview_token, symbol, name, market,
            strategy_type, signal_type, side, price, quantity, amount,
            confidence, final_grade, strategy_message, risk_approved,
            risk_code, risk_message, raw_pipeline_result, raw_strategy_decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            status,
            preview_token,
            req.symbol,
            req.name,
            req.market,
            req.strategy_type,
            strategy_decision["signal_type"],
            strategy_decision["side"],
            req.price,
            req.quantity,
            amount,
            strategy_decision["confidence"],
            strategy_decision["final_grade"],
            message,
            1 if risk_decision and risk_decision["approved"] else 0,
            risk_decision["code"] if risk_decision else None,
            risk_decision["message"] if risk_decision else None,
            json.dumps(pipeline_result, ensure_ascii=False),
            json.dumps(strategy_decision, ensure_ascii=False),
        ),
    )
    return int(cursor.lastrowid)


def _get_preview(conn: sqlite3.Connection, preview_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM order_previews
        WHERE id = ?
        """,
        (preview_id,),
    ).fetchone()


def _to_paper_request(
    req: OrderPreviewRequest,
    strategy_decision: dict[str, Any],
) -> PaperRunRequest:
    return PaperRunRequest(
        symbol=req.symbol,
        name=req.name,
        market=req.market,
        strategy_type=req.strategy_type,
        risk_level=req.risk_level,
        signal_type=strategy_decision["signal_type"],
        price=req.price,
        quantity=req.quantity,
        confidence=strategy_decision["confidence"],
        reason=strategy_decision["summary"],
        source="order_preview",
        **_paper_cost_kwargs(strategy_decision.get("cost_inputs") or {}),
    )


def _risk_decision_to_dict(decision: risk_manager.RiskDecision) -> dict[str, Any]:
    return {
        "approved": decision.approved,
        "code": decision.code,
        "message": decision.message,
        "checks": decision.checks,
    }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve_preview_quantity(
    req: OrderPreviewRequest,
) -> tuple[OrderPreviewRequest, dict[str, Any] | None]:
    recommendation = _preview_quantity_recommendation(req)
    if req.quantity is not None:
        if recommendation:
            recommendation = {
                **recommendation,
                "requested_quantity": req.quantity,
                "used_recommended_quantity": False,
            }
        return req, recommendation

    if recommendation is None:
        raise OrderApprovalError(
            "quantity is required unless account_equity, risk_per_trade, and stop_loss are provided",
            status_code=400,
        )

    recommended_quantity = int(recommendation["recommended_quantity"])
    if recommended_quantity <= 0:
        raise OrderApprovalError(
            "Recommended quantity is zero; check price, stop_loss, account_equity, risk_per_trade, and cash_available",
            status_code=400,
        )

    return (
        req.model_copy(update={"quantity": recommended_quantity}),
        {
            **recommendation,
            "requested_quantity": None,
            "used_recommended_quantity": True,
        },
    )


def _preview_quantity_recommendation(
    req: OrderPreviewRequest,
) -> dict[str, Any] | None:
    if (
        req.account_equity is None
        or req.risk_per_trade is None
        or req.stop_loss is None
    ):
        return None
    return risk_manager.recommend_order_quantity(
        price=req.price,
        stop_loss=req.stop_loss,
        account_equity=req.account_equity,
        risk_per_trade=req.risk_per_trade,
        cash_available=req.cash_available,
    )


def _cost_inputs_from_preview(
    req: OrderPreviewRequest,
    pipeline_result: dict[str, Any],
) -> dict[str, Any]:
    chart_flow = pipeline_result.get("chart_flow_result") or {}
    market_context = chart_flow.get("market_context") or {}
    intraday = (
        (chart_flow.get("price_data") or {}).get("intraday")
        or {}
    )
    spread_pct = intraday.get("spread_pct")
    inferred_spread_bps = spread_pct * 100 if isinstance(spread_pct, (int, float)) else None
    return {
        "risk_level": req.risk_level,
        "signal_time": req.signal_time,
        "decision_price": req.decision_price,
        "order_price": req.order_price,
        "signal_score": req.signal_score,
        "position_size": req.position_size,
        "stop_loss": req.stop_loss,
        "take_profit": req.take_profit,
        "trailing_stop": req.trailing_stop,
        "market_regime": req.market_regime or market_context.get("market_regime"),
        "model_version": req.model_version,
        "sector": req.sector,
        "account_equity": req.account_equity,
        "risk_per_trade": req.risk_per_trade,
        "cash_available": req.cash_available,
        "expected_gross_edge_bps": req.expected_gross_edge_bps,
        "expected_win_bps": req.expected_win_bps,
        "expected_loss_bps": req.expected_loss_bps,
        "expected_sharpe": req.expected_sharpe,
        "commission_rate": req.commission_rate,
        "sell_tax_rate": req.sell_tax_rate,
        "spread_bps": req.spread_bps if req.spread_bps is not None else inferred_spread_bps,
        "slippage_bps": req.slippage_bps,
        "fx_spread_bps": req.fx_spread_bps,
        "leverage": req.leverage,
        "margin_interest_rate": req.margin_interest_rate,
        "borrow_fee_rate": req.borrow_fee_rate,
        "expected_holding_days": req.expected_holding_days,
        "fill_probability": req.fill_probability,
        "is_short": req.is_short,
        "market_beta": req.market_beta,
    }


def _paper_cost_kwargs(cost_inputs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "expected_gross_edge_bps",
        "expected_win_bps",
        "expected_loss_bps",
        "expected_sharpe",
        "commission_rate",
        "sell_tax_rate",
        "spread_bps",
        "slippage_bps",
        "fx_spread_bps",
        "leverage",
        "margin_interest_rate",
        "borrow_fee_rate",
        "expected_holding_days",
        "fill_probability",
        "is_short",
        "market_beta",
        "signal_time",
        "decision_price",
        "order_price",
        "signal_score",
        "position_size",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "market_regime",
        "model_version",
        "sector",
        "account_equity",
        "risk_per_trade",
        "cash_available",
    }
    return {key: value for key, value in cost_inputs.items() if key in allowed}
