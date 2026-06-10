from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from app.brokers.kis_client import KIS_PAPER_BASE_URL, KisClient
from app.config import settings
from app.models import LiveOrderRequest, PaperRunRequest
from app.storage.market_data import get_latest_market_context
from app.trading import broker_sync
from app.trading import order_state
from app.trading import paper_trading, risk_manager


class LiveTradingError(ValueError):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details or {}


def execute_live_order(
    req: LiveOrderRequest,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute a live KIS limit order after explicit safety checks."""
    _validate_live_trading_gate(req)

    resolved_db_path = settings.storage_path(db_path or paper_trading.DEFAULT_DB_PATH)
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


def execute_broker_paper_order(
    req: LiveOrderRequest,
    db_path: Path | str | None = None,
    *,
    client: KisClient | None = None,
) -> dict[str, Any]:
    """Submit a limit order to the KIS mock trading account."""
    client = client or KisClient(is_paper=True)
    safety = broker_paper_safety_check(req=req, client=client)
    if not safety.get("approved"):
        reason = str(
            safety.get("broker_submit_block_reason")
            or "Broker paper submit blocked"
        )
        raise LiveTradingError(
            reason,
            status_code=403,
            code=str(safety.get("broker_submit_block_code") or reason),
            details=safety,
        )

    resolved_db_path = settings.storage_path(db_path or paper_trading.DEFAULT_DB_PATH)
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

    guard = order_state.validate_broker_paper_order(req)
    if not guard.get("approved"):
        reason = str(
            guard.get("broker_submit_block_reason")
            or "Broker paper order blocked"
        )
        raise LiveTradingError(
            reason,
            status_code=403,
            code=guard.get("broker_submit_block_code"),
            details=guard,
        )

    try:
        intent = order_state.begin_order_intent(req)
    except order_state.OrderStateError as exc:
        raise LiveTradingError(
            str(exc),
            status_code=exc.status_code,
            code=exc.code,
        ) from exc

    try:
        kis_result = _submit_with_limited_retries(client, req)
    except Exception as exc:
        failed_intent = order_state.mark_order_failed(intent["id"], str(exc))
        event = _record_broker_paper_event(
            req=req,
            order_status="rejected",
            reject_reason=str(exc),
            raw_response=_exception_payload(exc),
        )
        return {
            "status": "rejected",
            "message": f"KIS mock order rejected or failed: {exc}",
            "symbol": req.symbol,
            "side": req.side,
            "order_type": req.order_type,
            "price": req.price,
            "quantity": req.quantity,
            "kis_result": None,
            "order_state": failed_intent,
            "order_event": event,
            "broker_submit_blocked": False,
        }

    order_status = _broker_order_status_from_response(kis_result)
    if order_status == "rejected":
        failed_intent = order_state.mark_order_failed(
            intent["id"],
            str(kis_result.get("msg1") or kis_result.get("message") or "KIS rejected order"),
        )
        event = _record_broker_paper_event(
            req=req,
            order_status="rejected",
            raw_response=kis_result,
            reject_reason=str(
                kis_result.get("msg1")
                or kis_result.get("message")
                or "KIS rejected order"
            ),
        )
        return {
            "status": "rejected",
            "message": str(
                kis_result.get("msg1")
                or kis_result.get("message")
                or "KIS mock order rejected"
            ),
            "symbol": req.symbol,
            "side": req.side,
            "order_type": req.order_type,
            "price": req.price,
            "quantity": req.quantity,
            "kis_result": kis_result,
            "order_state": failed_intent,
            "order_event": event,
            "broker_submit_blocked": False,
        }
    event = _record_broker_paper_event(
        req=req,
        order_status=order_status,
        raw_response=kis_result,
    )
    intent = order_state.mark_order_submitted(intent["id"], kis_result)
    broker_sync_result = _sync_after_live_order(client)
    order_state_result = _reconcile_order_state(
        req=req,
        intent=intent,
        broker_sync_result=broker_sync_result,
    )
    return {
        "status": order_status,
        "message": "KIS mock limit order submitted",
        "symbol": req.symbol,
        "side": req.side,
        "order_type": req.order_type,
        "price": req.price,
        "quantity": req.quantity,
        "kis_result": kis_result,
        "broker_sync": broker_sync_result,
        "order_state": order_state_result,
        "order_event": event,
        "broker_submit_blocked": False,
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


def broker_paper_safety_check(
    *,
    req: LiveOrderRequest | None = None,
    client: KisClient | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Return whether broker_paper mode is safe to submit to KIS mock trading."""
    provider = str(getattr(req, "broker_provider", "kis") or "kis").lower()
    if provider != "kis":
        return _broker_submit_block(f"broker_provider must be kis, got {provider}")
    if not settings.kis_is_paper:
        return _broker_submit_block("broker_paper requires KIS_IS_PAPER=true")
    storage = settings.storage_status()
    if not storage.get("data_dir_writable") or not storage.get("data_dir_is_persistent"):
        return _broker_submit_block(
            "persistent_order_storage_unavailable",
            {
                "resolved_data_dir": storage.get("resolved_data_dir"),
                "data_dir_writable": storage.get("data_dir_writable"),
                "data_dir_is_persistent": storage.get("data_dir_is_persistent"),
                "data_dir_warning": storage.get("data_dir_warning"),
                "storage_root_fallback_used": storage.get("storage_root_fallback_used"),
            },
        )

    client = client or KisClient(is_paper=True)
    diagnostics = (
        client.runtime_diagnostics()
        if hasattr(client, "runtime_diagnostics")
        else {
            "base_url": getattr(client, "base_url", ""),
            "is_paper": getattr(client, "is_paper", None),
            "app_key_configured": bool(getattr(client, "app_key", None)),
            "app_secret_configured": bool(getattr(client, "app_secret", None)),
            "account_no_configured": bool(getattr(client, "account_no", None)),
            "account_product_code": getattr(client, "account_product_code", "") or "",
        }
    )
    base_url = str(diagnostics.get("base_url") or getattr(client, "base_url", ""))
    if not bool(diagnostics.get("is_paper", getattr(client, "is_paper", False))):
        return _broker_submit_block("KIS client is not configured for paper trading", diagnostics)
    if base_url.rstrip("/") != KIS_PAPER_BASE_URL:
        return _broker_submit_block("KIS order endpoint is not the mock trading endpoint", diagnostics)
    for key in ("app_key_configured", "app_secret_configured", "account_no_configured"):
        if diagnostics.get(key) is False:
            return _broker_submit_block(f"KIS mock account configuration missing: {key}", diagnostics)
    if not diagnostics.get("account_product_code"):
        return _broker_submit_block("KIS mock account product code is missing", diagnostics)
    if (
        diagnostics.get("kis_token_refresh_blocked_by_rate_limit")
        and not diagnostics.get("kis_token_cached")
    ):
        return _broker_submit_block(
            "kis_token_unavailable_rate_limited",
            diagnostics,
            code="kis_token_unavailable_rate_limited",
        )

    probe_symbol = symbol or getattr(req, "symbol", None) or "005930"
    try:
        quote = client.get_current_price(probe_symbol)
        price = _extract_current_price(quote)
    except Exception as exc:
        return _broker_submit_block(f"KIS mock quote connectivity failed: {exc}", diagnostics)
    if price <= 0:
        return _broker_submit_block("KIS mock quote connectivity returned no positive price", diagnostics)

    try:
        balance = client.get_balance()
        cash = _extract_cash_or_buying_power(balance)
    except Exception as exc:
        return _broker_submit_block(f"KIS mock account probe failed: {exc}", diagnostics)
    if cash is None:
        return _broker_submit_block("KIS mock cash/buying_power is unavailable", diagnostics)
    if cash <= 0:
        return _broker_submit_block(
            "KIS mock cash/buying_power is zero",
            diagnostics,
            code="cash_or_buying_power_zero",
        )

    return {
        "approved": True,
        "broker_submit_blocked": False,
        "broker_submit_block_reason": None,
        "broker_provider": "kis",
        "kis_is_paper": True,
        "submits_to_broker": True,
        "quote_connected": True,
        "balance_connected": True,
        "cash_or_buying_power": cash,
        "current_price": price,
        "diagnostics": diagnostics,
    }


def _broker_submit_block(
    reason: str,
    diagnostics: dict[str, Any] | None = None,
    *,
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "approved": False,
        "broker_submit_blocked": True,
        "broker_submit_block_reason": reason,
        "broker_submit_block_code": code or reason,
        "broker_provider": "kis",
        "kis_is_paper": bool(settings.kis_is_paper),
        "submits_to_broker": False,
        "diagnostics": diagnostics or {},
    }


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


def _record_broker_paper_event(
    *,
    req: LiveOrderRequest,
    order_status: str,
    raw_response: dict[str, Any],
    reject_reason: str | None = None,
) -> dict[str, Any]:
    return order_state.record_broker_order_event(
        {
            "symbol": req.symbol,
            "name": getattr(req, "name", None),
            "session_id": req.session_id,
            "scan_id": req.scan_id,
            "side": req.side,
            "qty": req.quantity,
            "order_type": req.order_type,
            "limit_price": req.price,
            "submitted_price": req.price,
            "notional_krw": float(req.price) * int(req.quantity),
            "broker_provider": "kis",
            "kis_is_paper": True,
            "execution_mode": "broker_paper",
            "broker_order_id": _extract_order_no(raw_response),
            "broker_response_code": raw_response.get("rt_cd") or raw_response.get("msg_cd"),
            "broker_response_message": raw_response.get("msg1") or raw_response.get("message"),
            "order_status": order_status,
            "reject_reason": reject_reason,
            "raw_response": raw_response,
        }
    )


def _broker_order_status_from_response(response: dict[str, Any]) -> str:
    rt_cd = str(response.get("rt_cd") or response.get("rtcode") or "").strip()
    if rt_cd and rt_cd != "0":
        return "rejected"
    if _extract_order_no(response):
        return "submitted"
    return "unknown_pending"


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


def _exception_payload(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
        "error_code": getattr(exc, "error_code", None),
        "error_description": getattr(exc, "error_description", None),
    }


def _extract_current_price(quote: dict[str, Any]) -> float:
    output = quote.get("output") if isinstance(quote, dict) else {}
    if not isinstance(output, dict):
        output = {}
    for key in ("stck_prpr", "prpr", "last", "price"):
        value = _to_float(output.get(key) or quote.get(key))
        if value and value > 0:
            return value
    return 0.0


def _extract_cash_or_buying_power(balance: dict[str, Any]) -> float | None:
    for row in _rows(balance, "output2", "output"):
        value = _pick_float(
            row,
            "dnca_tot_amt",
            "ord_psbl_cash",
            "ord_psbl_amt",
            "buying_power",
            "cash",
            "nass_amt",
            "tot_evlu_amt",
        )
        if value is not None:
            return value
    return _pick_float(balance, "cash", "buying_power", "total_cash", "total_value")


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


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


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
        strategy_type=req.strategy_type,
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
        trailing_stop=req.trailing_stop,
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
