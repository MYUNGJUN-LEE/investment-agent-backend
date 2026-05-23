from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from app.brokers.kis_client import KisClient
from app.config import settings
from app.models import LiveOrderRequest, PaperRunRequest
from app.storage.market_data import get_latest_market_context
from app.trading import broker_sync
from app.trading import order_state
from app.trading import paper_trading, risk_manager


class LiveTradingError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def execute_live_order(
    req: LiveOrderRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute a live KIS limit order after explicit safety checks."""
    _validate_live_trading_gate(req)

    resolved_db_path = Path(db_path) if db_path else paper_trading.DEFAULT_DB_PATH
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_db_path) as conn:
        conn.row_factory = sqlite3.Row
        paper_trading.initialize_db(conn)
        paper_req = _to_paper_risk_request(req)
        decision = risk_manager.approve_order(
            conn=conn,
            req=paper_req,
            side="BUY" if req.side == "buy" else "SELL",
            quantity=req.quantity,
        )

    if not decision.approved:
        raise LiveTradingError(
            f"Risk rejected: {decision.message} ({decision.code})",
            status_code=403,
        )

    client = KisClient(is_paper=False)
    try:
        intent = order_state.begin_order_intent(req)
    except order_state.OrderStateError as exc:
        raise LiveTradingError(str(exc), status_code=exc.status_code) from exc

    try:
        kis_result = _submit_with_limited_retries(client, req)
    except Exception as exc:
        order_state.mark_order_failed(intent["id"], str(exc))
        raise

    intent = order_state.mark_order_submitted(intent["id"], kis_result)
    broker_sync_result = _sync_after_live_order(client)
    order_state_result = _reconcile_order_state(
        req=req,
        intent=intent,
        broker_sync_result=broker_sync_result,
    )
    return {
        "status": "submitted",
        "message": "Live limit order submitted to KIS",
        "symbol": req.symbol,
        "side": req.side,
        "order_type": req.order_type,
        "price": req.price,
        "quantity": req.quantity,
        "kis_result": kis_result,
        "broker_sync": broker_sync_result,
        "order_state": order_state_result,
    }


def _validate_live_trading_gate(req: LiveOrderRequest) -> None:
    if not settings.enable_live_trading:
        raise LiveTradingError("Live trading is disabled", status_code=403)
    if settings.kis_is_paper:
        raise LiveTradingError(
            "Live orders require KIS_IS_PAPER=false",
            status_code=403,
        )
    if not settings.live_trading_confirm_token:
        raise LiveTradingError(
            "LIVE_TRADING_CONFIRM_TOKEN is not configured",
            status_code=403,
        )
    if req.confirm_token != settings.live_trading_confirm_token:
        raise LiveTradingError("Invalid live trading confirm_token", status_code=403)
    if req.order_type != "limit":
        raise LiveTradingError("Only limit orders are allowed", status_code=400)


def _submit_with_limited_retries(
    client: KisClient,
    req: LiveOrderRequest,
) -> dict[str, Any]:
    attempts = max(1, int(settings.max_order_api_retries))
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return client.place_domestic_limit_order(
                symbol=req.symbol,
                side=req.side,
                price=req.price,
                quantity=req.quantity,
            )
        except Exception as exc:
            last_error = exc
    raise LiveTradingError(
        f"KIS order failed after {attempts} attempt(s): {last_error}",
        status_code=502,
    )


def _sync_after_live_order(client: KisClient) -> dict[str, Any] | None:
    try:
        return broker_sync.sync_kis_account(client=client)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Broker sync failed after live order: {exc}",
        }


def _reconcile_order_state(
    req: LiveOrderRequest,
    intent: dict[str, Any],
    broker_sync_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if broker_sync_result and broker_sync_result.get("status") == "success":
        account_no = str(broker_sync_result.get("account_no") or "")
        return order_state.reconcile_after_broker_sync(
            symbol=req.symbol,
            market=req.market,
            account_no=account_no,
        )
    return {
        "intent_id": intent.get("id"),
        "symbol": req.symbol,
        "state": intent.get("position_state_after"),
        "status": intent.get("status"),
        "message": "Broker sync did not complete; position remains pending",
    }


def _to_paper_risk_request(req: LiveOrderRequest) -> PaperRunRequest:
    latest_context = get_latest_market_context() or {}
    return PaperRunRequest(
        symbol=req.symbol,
        market=req.market,
        strategy_type="daytrade",
        risk_level=req.risk_level,
        signal_type="entry" if req.side == "buy" else "exit",
        price=req.price,
        quantity=req.quantity,
        confidence=1.0,
        reason=req.reason,
        source="live_order",
        signal_time=req.signal_time,
        decision_price=req.decision_price,
        order_price=req.order_price,
        signal_score=req.signal_score,
        position_size=req.position_size,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        market_regime=req.market_regime or latest_context.get("market_regime"),
        model_version=req.model_version,
        sector=req.sector,
        account_equity=req.account_equity,
        risk_per_trade=req.risk_per_trade,
        cash_available=req.cash_available,
        expected_gross_edge_bps=req.expected_gross_edge_bps,
        expected_win_bps=req.expected_win_bps,
        expected_loss_bps=req.expected_loss_bps,
        expected_sharpe=req.expected_sharpe,
        commission_rate=req.commission_rate,
        sell_tax_rate=req.sell_tax_rate,
        spread_bps=req.spread_bps,
        slippage_bps=req.slippage_bps,
        fx_spread_bps=req.fx_spread_bps,
        leverage=req.leverage,
        margin_interest_rate=req.margin_interest_rate,
        borrow_fee_rate=req.borrow_fee_rate,
        expected_holding_days=req.expected_holding_days,
        fill_probability=req.fill_probability,
        is_short=req.is_short,
        market_beta=req.market_beta,
    )
