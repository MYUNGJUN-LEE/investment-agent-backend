from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any
from uuid import uuid4

from app.config import settings
from app.data_sources.kis import fetch_price_data
from app.models import (
    AutoTradeStartRequest,
    AutoTradeSymbolConfig,
    GptAutoTradeControlRequest,
    LiveOrderRequest,
    OrderConfirmRequest,
    OrderPreviewRequest,
)
from app.trading.live_trading import LiveTradingError, execute_live_order
from app.trading.order_approval import (
    OrderApprovalError,
    confirm_order_preview,
    create_order_preview,
)
from app.trading import auto_trading_store
from app.trading import broker_sync
from app.trading import paper_trading
from app.trading import risk_manager
from app.trading.universe_scanner import scan_universe_for_auto_trade


class AutoTradingError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def start_auto_trading(req: AutoTradeStartRequest) -> dict[str, Any]:
    """Persist an auto-trading session for a separate worker process."""
    _validate_start_request(req)
    session = auto_trading_store.create_session(req)
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "execution_mode": req.execution_mode,
        "interval_seconds": req.interval_seconds,
        "max_cycles": req.max_cycles,
        "started_at": session["created_at"],
        "message": "Auto-trading session saved. Run the auto-trading worker to process it.",
        "universe_scan": None,
    }


def get_auto_trading_status(session_id: str) -> dict[str, Any]:
    session = auto_trading_store.get_session(session_id)
    if not session:
        return {
            "session_id": session_id,
            "status": "not_found",
            "message": "Auto-trading session not found",
        }
    return auto_trading_store.session_to_status(session)


def list_auto_trading_sessions(
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    sessions = [
        auto_trading_store.session_to_status(session)
        for session in auto_trading_store.list_sessions(status=status, limit=limit)
    ]
    return {
        "count": len(sessions),
        "sessions": sessions,
    }


def list_auto_trading_events(session_id: str, limit: int = 100) -> dict[str, Any]:
    events = auto_trading_store.list_events(session_id=session_id, limit=limit)
    return {
        "session_id": session_id,
        "count": len(events),
        "events": events,
    }


def stop_auto_trading(session_id: str) -> dict[str, Any]:
    session = auto_trading_store.stop_session(session_id)
    if not session:
        return {
            "session_id": session_id,
            "status": "not_found",
            "message": "Auto-trading session not found",
        }
    return {
        "session_id": session_id,
        "status": session["status"],
        "message": "Auto-trading session stopped",
    }


def control_auto_trading_from_gpt(req: GptAutoTradeControlRequest) -> dict[str, Any]:
    """Simple Custom GPT control surface: start, stop, or status."""
    worker_status = _embedded_worker_status()
    active = [
        auto_trading_store.session_to_status(session)
        for session in auto_trading_store.list_sessions(status="active", limit=500)
    ]
    if req.command == "status":
        return {
            "status": "success",
            "command": req.command,
            "message": f"{len(active)} active auto-trading session(s)",
            "active_sessions": active,
            "recent_sessions": list_auto_trading_sessions(limit=10)["sessions"],
            "stopped_sessions": [],
            "started_session": None,
            "worker_status": worker_status,
        }

    if req.command == "stop":
        stopped = []
        for session in active:
            stopped.append(stop_auto_trading(session["session_id"]))
        return {
            "status": "success",
            "command": req.command,
            "message": f"Stopped {len(stopped)} active auto-trading session(s)",
            "active_sessions": [],
            "recent_sessions": list_auto_trading_sessions(limit=10)["sessions"],
            "stopped_sessions": stopped,
            "started_session": None,
            "worker_status": worker_status,
        }

    if active and not req.force_new:
        return {
            "status": "already_active",
            "command": req.command,
            "message": "Auto-trading is already active; no duplicate session created",
            "active_sessions": active,
            "recent_sessions": list_auto_trading_sessions(limit=10)["sessions"],
            "stopped_sessions": [],
            "started_session": None,
            "worker_status": worker_status,
        }

    worker_status = _start_embedded_workers_if_enabled()
    start_req = AutoTradeStartRequest(
        symbols=[],
        auto_discover_symbols=req.auto_discover_symbols,
        universe_seed_symbols=req.universe_seed_symbols,
        universe_candidate_limit=req.universe_candidate_limit,
        universe_final_limit=req.universe_final_limit,
        execution_mode=req.execution_mode,
        interval_seconds=req.interval_seconds,
        max_cycles=req.max_cycles,
        run_immediately=True,
        auto_confirm_paper=req.auto_confirm_paper,
        account_equity=req.account_equity,
        risk_per_trade=req.risk_per_trade,
        cash_available=req.cash_available,
        live_confirm_token=req.live_confirm_token,
    )
    started = start_auto_trading(start_req)
    return {
        "status": "started",
        "command": req.command,
        "message": "Auto-trading started with universe auto-discovery",
        "active_sessions": list_auto_trading_sessions(status="active", limit=10)["sessions"],
        "recent_sessions": list_auto_trading_sessions(limit=10)["sessions"],
        "stopped_sessions": [],
        "started_session": started,
        "worker_status": worker_status,
    }


def restart_auto_trading(session_id: str) -> dict[str, Any]:
    session = auto_trading_store.get_session(session_id)
    if not session:
        return {
            "session_id": session_id,
            "status": "not_found",
            "message": "Auto-trading session not found",
        }
    _validate_start_request(auto_trading_store.load_request(session))
    restarted = auto_trading_store.restart_session(session_id)
    return {
        "session_id": session_id,
        "status": (restarted or {}).get("status", "unknown"),
        "message": "Auto-trading session restarted",
    }


def run_auto_trading_once(req: AutoTradeStartRequest) -> dict[str, Any]:
    """Run one auto-trading cycle synchronously for tests or one-shot operation."""
    _validate_start_request(req)
    return {
        "execution_mode": req.execution_mode,
        "results": _run_cycle(req),
    }


def process_due_sessions(
    worker_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim and process due persistent sessions once."""
    worker_id = worker_id or f"auto-worker-{uuid4().hex[:8]}"
    sessions = auto_trading_store.claim_due_sessions(worker_id=worker_id, limit=limit)
    processed: list[dict[str, Any]] = []
    for session in sessions:
        try:
            req = auto_trading_store.load_request(session)
            _validate_start_request(req)
            results = _run_cycle(req, session_id=session["session_id"])
            sync_result = _sync_live_account_if_needed(req)
            if sync_result is not None:
                results.append(
                    {
                        "symbol": "__account__",
                        "status": sync_result.get("status", "unknown"),
                        "broker_sync": sync_result,
                    }
                )
            updated = auto_trading_store.complete_cycle(session["session_id"], results)
            processed.append(
                {
                    "session_id": session["session_id"],
                    "status": (updated or {}).get("status", "unknown"),
                    "results": results,
                }
            )
        except Exception as exc:
            updated = auto_trading_store.fail_cycle(session["session_id"], str(exc))
            processed.append(
                {
                    "session_id": session["session_id"],
                    "status": (updated or {}).get("status", "error"),
                    "error": str(exc),
                }
            )
    return processed


def run_worker_forever(
    worker_id: str | None = None,
    poll_seconds: float | None = None,
) -> None:
    """Run a simple APScheduler/RQ-style polling worker process."""
    worker_id = worker_id or f"auto-worker-{uuid4().hex[:8]}"
    poll_seconds = (
        settings.auto_trading_worker_poll_seconds
        if poll_seconds is None
        else poll_seconds
    )
    auto_trading_store.initialize_auto_trading_db()
    while True:
        process_due_sessions(worker_id=worker_id)
        time.sleep(float(poll_seconds))


def _run_cycle(
    req: AutoTradeStartRequest,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    symbols = list(req.symbols)
    results_prefix: list[dict[str, Any]] = []
    if req.auto_discover_symbols and not symbols:
        scan = scan_universe_for_auto_trade(req)
        symbols = scan["symbols"]
        results_prefix.append(
            {
                "symbol": "__universe__",
                "status": scan["status"],
                "scan_id": scan["scan_id"],
                "source_symbol_count": scan["source_symbol_count"],
                "snapshot_count": scan.get("snapshot_count", scan["source_symbol_count"]),
                "candidate_count": scan["candidate_count"],
                "final_count": scan["final_count"],
                "final_candidates": scan["final_candidates"],
            }
        )
        min_scanned_symbols = max(
            0,
            int(settings.universe_scanner_min_scanned_symbols_for_trading or 0),
        )
        scanned_symbols = int(
            scan.get("snapshot_count") or scan.get("source_symbol_count") or 0
        )
        if min_scanned_symbols and scanned_symbols < min_scanned_symbols:
            results_prefix[0]["status"] = "blocked"
            results_prefix[0]["message"] = (
                f"Universe scanner scanned {scanned_symbols} symbols; "
                f"at least {min_scanned_symbols} symbols are required before trading"
            )
            return results_prefix
        if not symbols:
            results_prefix[0]["message"] = "Universe scanner found no tradable candidates"
            return results_prefix

    prepared = _prepare_symbols_for_account_balance(req, symbols)
    results_prefix.extend(prepared["results"])
    symbols = prepared["symbols"]
    if not symbols:
        return results_prefix

    if len(symbols) <= 1:
        return results_prefix + [
            _run_symbol(req, symbol_cfg, session_id=session_id) for symbol_cfg in symbols
        ]

    max_workers = max(
        1,
        min(len(symbols), int(settings.auto_trading_symbol_workers or 1)),
    )
    results: list[dict[str, Any] | None] = [None] * len(symbols)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_symbol, req, symbol_cfg, session_id): index
            for index, symbol_cfg in enumerate(symbols)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results_prefix + [result for result in results if result is not None]


def _sync_live_account_if_needed(req: AutoTradeStartRequest) -> dict[str, Any] | None:
    if req.execution_mode != "live":
        return None
    try:
        return broker_sync.sync_kis_account()
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Broker sync failed during auto-trading cycle: {exc}",
        }


def _prepare_symbols_for_account_balance(
    req: AutoTradeStartRequest,
    symbols: list[AutoTradeSymbolConfig],
) -> dict[str, Any]:
    priced_symbols: list[AutoTradeSymbolConfig] = []
    blocked_results: list[dict[str, Any]] = []
    for symbol_cfg in symbols:
        price_result = _resolve_loop_price(symbol_cfg)
        if not price_result.get("price"):
            blocked_results.append(
                {
                    "symbol": symbol_cfg.symbol,
                    "status": "blocked",
                    "message": price_result["message"],
                    "price_source": price_result["source"],
                }
            )
            continue
        price = float(price_result["price"])
        priced_symbols.append(
            _apply_order_sizing_defaults(
                req,
                symbol_cfg.model_copy(
                    update={
                        "price": symbol_cfg.price or price,
                        "decision_price": symbol_cfg.decision_price or price,
                        "order_price": symbol_cfg.order_price or price,
                    }
                ),
                price,
            )
        )

    if not priced_symbols:
        return {"symbols": [], "results": blocked_results}

    account = _resolve_account_balance(req)
    if account["status"] != "ready":
        return {
            "symbols": [],
            "results": [
                *blocked_results,
                {
                    "symbol": "__account__",
                    "status": "blocked",
                    "message": account["message"],
                    "account": account,
                },
            ],
        }

    allocated = _allocate_cash_to_symbols(priced_symbols, account)
    return {
        "symbols": allocated["symbols"],
        "results": [*blocked_results, *allocated["blocked_results"]],
    }


def _resolve_account_balance(req: AutoTradeStartRequest) -> dict[str, Any]:
    if req.execution_mode == "live":
        try:
            sync_result = broker_sync.sync_kis_account()
        except Exception as exc:
            return {
                "status": "blocked",
                "mode": "live",
                "message": f"Live account balance check failed; no orders will be attempted: {exc}",
            }
        cash_available = _to_float(sync_result.get("total_cash"))
        account_equity = _to_float(sync_result.get("total_value")) or req.account_equity
        if cash_available is None:
            return {
                "status": "blocked",
                "mode": "live",
                "message": "Live account cash balance is unavailable; no orders will be attempted.",
                "broker_sync": sync_result,
            }
        return {
            "status": "ready",
            "mode": "live",
            "account_equity": float(account_equity),
            "cash_available": max(0.0, float(cash_available)),
            "broker_sync": sync_result,
        }

    snapshot = paper_trading.get_paper_account_snapshot(
        account_equity=req.account_equity,
    )
    cash_available = float(snapshot["cash_available"])
    if req.cash_available is not None:
        cash_available = min(cash_available, float(req.cash_available))
    return {
        "status": "ready",
        "mode": "paper",
        "account_equity": float(snapshot["account_equity"]),
        "cash_available": max(0.0, cash_available),
        "paper_account": snapshot,
    }


def _allocate_cash_to_symbols(
    symbols: list[AutoTradeSymbolConfig],
    account: dict[str, Any],
) -> dict[str, Any]:
    remaining_cash = float(account["cash_available"])
    account_equity = float(account["account_equity"])
    blocked_by_index: dict[int, dict[str, Any]] = {}
    allocated_by_index: dict[int, AutoTradeSymbolConfig] = {}
    order = sorted(
        range(len(symbols)),
        key=lambda index: float(symbols[index].signal_score or 0),
        reverse=True,
    )

    for index in order:
        symbol_cfg = symbols[index]
        if symbol_cfg.requested_action == "exit":
            allocated_by_index[index] = symbol_cfg
            continue

        price = float(symbol_cfg.price or symbol_cfg.order_price or 0)
        if price <= 0:
            blocked_by_index[index] = _insufficient_cash_result(
                symbol_cfg=symbol_cfg,
                available_cash=remaining_cash,
                account=account,
                reason="missing price",
            )
            continue

        requested_quantity = symbol_cfg.quantity
        if requested_quantity is not None:
            required_cash = price * int(requested_quantity)
            if required_cash > remaining_cash:
                blocked_by_index[index] = _insufficient_cash_result(
                    symbol_cfg=symbol_cfg,
                    available_cash=remaining_cash,
                    account=account,
                    reason=f"required {required_cash:.2f}",
                )
                continue
            allocation = required_cash
        else:
            if symbol_cfg.stop_loss is None:
                blocked_by_index[index] = _insufficient_cash_result(
                    symbol_cfg=symbol_cfg,
                    available_cash=remaining_cash,
                    account=account,
                    reason="missing stop_loss",
                )
                continue
            recommendation = risk_manager.recommend_order_quantity(
                price=price,
                stop_loss=float(symbol_cfg.stop_loss),
                account_equity=account_equity,
                risk_per_trade=float(symbol_cfg.risk_per_trade or 0.005),
                cash_available=remaining_cash,
            )
            recommended_quantity = int(recommendation["recommended_quantity"])
            if recommended_quantity <= 0:
                blocked_by_index[index] = _insufficient_cash_result(
                    symbol_cfg=symbol_cfg,
                    available_cash=remaining_cash,
                    account=account,
                    reason="available cash is below the minimum executable quantity",
                    recommendation=recommendation,
                )
                continue
            allocation = recommended_quantity * price

        remaining_cash = max(0.0, remaining_cash - allocation)
        allocated_by_index[index] = symbol_cfg.model_copy(
            update={
                "account_equity": account_equity,
                "cash_available": allocation,
            }
        )

    return {
        "symbols": [
            allocated_by_index[index]
            for index in range(len(symbols))
            if index in allocated_by_index
        ],
        "blocked_results": [
            blocked_by_index[index]
            for index in range(len(symbols))
            if index in blocked_by_index
        ],
    }


def _insufficient_cash_result(
    *,
    symbol_cfg: AutoTradeSymbolConfig,
    available_cash: float,
    account: dict[str, Any],
    reason: str,
    recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol_cfg.symbol,
        "status": "blocked",
        "message": (
            "Insufficient available cash; no order preview or broker order was attempted "
            f"({reason})."
        ),
        "account": {
            "mode": account.get("mode"),
            "account_equity": account.get("account_equity"),
            "cash_available": round(float(available_cash), 2),
        },
        "recommendation": recommendation,
    }


def _run_symbol(
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
    session_id: str | None = None,
) -> dict[str, Any]:
    try:
        price_result = _resolve_loop_price(symbol_cfg)
        if not price_result.get("price"):
            return {
                "symbol": symbol_cfg.symbol,
                "status": "blocked",
                "message": price_result["message"],
                "price_source": price_result["source"],
            }

        price = float(price_result["price"])
        symbol_cfg = _apply_order_sizing_defaults(req, symbol_cfg, price)
        preview_req = _to_preview_request(symbol_cfg, price)
        preview = create_order_preview(preview_req)
        result: dict[str, Any] = {
            "symbol": symbol_cfg.symbol,
            "status": preview["status"],
            "price": price,
            "price_source": price_result["source"],
            "preview": preview,
        }
        if preview["status"] != "pending":
            result["message"] = preview["message"]
            return result

        if req.execution_mode == "paper":
            if not req.auto_confirm_paper:
                result["message"] = "Paper preview created but auto_confirm_paper is false"
                return result
            result["execution"] = confirm_order_preview(
                OrderConfirmRequest(
                    preview_id=preview["preview_id"],
                    preview_token=preview["preview_token"] or "",
                    execution_mode="paper",
                )
            )
            result["status"] = result["execution"]["status"]
            return result

        result["execution"] = execute_live_order(
            _to_live_order_request(
                req=req,
                symbol_cfg=symbol_cfg,
                preview=preview,
                price=price,
                session_id=session_id,
            )
        )
        result["status"] = result["execution"]["status"]
        return result
    except (OrderApprovalError, LiveTradingError, AutoTradingError) as exc:
        return {
            "symbol": symbol_cfg.symbol,
            "status": "error",
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "symbol": symbol_cfg.symbol,
            "status": "error",
            "message": f"Unexpected auto-trading error: {exc}",
        }


def _to_preview_request(
    symbol_cfg: AutoTradeSymbolConfig,
    price: float,
) -> OrderPreviewRequest:
    return OrderPreviewRequest(
        symbol=symbol_cfg.symbol,
        name=symbol_cfg.name,
        market=symbol_cfg.market,
        strategy_type=symbol_cfg.strategy_type,
        lookback_hours=symbol_cfg.lookback_hours,
        risk_level=symbol_cfg.risk_level,
        requested_action=symbol_cfg.requested_action,
        price=price,
        quantity=symbol_cfg.quantity,
        decision_price=symbol_cfg.decision_price or price,
        order_price=symbol_cfg.order_price or price,
        signal_score=symbol_cfg.signal_score,
        position_size=symbol_cfg.position_size,
        stop_loss=symbol_cfg.stop_loss,
        take_profit=symbol_cfg.take_profit,
        market_regime=symbol_cfg.market_regime,
        model_version=symbol_cfg.model_version,
        sector=symbol_cfg.sector,
        account_equity=symbol_cfg.account_equity,
        risk_per_trade=symbol_cfg.risk_per_trade,
        cash_available=symbol_cfg.cash_available,
        expected_gross_edge_bps=symbol_cfg.expected_gross_edge_bps,
        expected_win_bps=symbol_cfg.expected_win_bps,
        expected_loss_bps=symbol_cfg.expected_loss_bps,
        expected_sharpe=symbol_cfg.expected_sharpe,
        commission_rate=symbol_cfg.commission_rate,
        sell_tax_rate=symbol_cfg.sell_tax_rate,
        spread_bps=symbol_cfg.spread_bps,
        slippage_bps=symbol_cfg.slippage_bps,
        fx_spread_bps=symbol_cfg.fx_spread_bps,
        leverage=symbol_cfg.leverage,
        margin_interest_rate=symbol_cfg.margin_interest_rate,
        borrow_fee_rate=symbol_cfg.borrow_fee_rate,
        expected_holding_days=symbol_cfg.expected_holding_days,
        fill_probability=symbol_cfg.fill_probability,
        is_short=symbol_cfg.is_short,
        market_beta=symbol_cfg.market_beta,
    )


def _apply_order_sizing_defaults(
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
    price: float,
) -> AutoTradeSymbolConfig:
    updates: dict[str, Any] = {}
    if symbol_cfg.account_equity is None:
        updates["account_equity"] = req.account_equity
    if symbol_cfg.risk_per_trade is None:
        updates["risk_per_trade"] = req.risk_per_trade
    if symbol_cfg.cash_available is None and req.cash_available is not None:
        updates["cash_available"] = req.cash_available
    if symbol_cfg.stop_loss is None and symbol_cfg.quantity is None:
        updates["stop_loss"] = price * (1 - settings.monitor_default_stop_loss_pct / 100)
    return symbol_cfg.model_copy(update=updates) if updates else symbol_cfg


def _to_live_order_request(
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
    preview: dict[str, Any],
    price: float,
    session_id: str | None = None,
) -> LiveOrderRequest:
    if not req.live_confirm_token:
        raise AutoTradingError("live_confirm_token is required for live auto-trading")
    side = str(preview["side"]).lower()
    return LiveOrderRequest(
        symbol=symbol_cfg.symbol,
        market=symbol_cfg.market,
        risk_level=symbol_cfg.risk_level,
        side=side,
        order_type="limit",
        price=float(preview["price"]),
        quantity=int(preview["quantity"]),
        confirm_token=req.live_confirm_token,
        client_order_id=_auto_client_order_id(
            session_id=session_id,
            symbol=symbol_cfg.symbol,
            side=side,
            price=float(preview["price"]),
            quantity=int(preview["quantity"]),
        ),
        session_id=session_id,
        reason=f"auto-trading {preview.get('message')}",
        decision_price=symbol_cfg.decision_price or price,
        order_price=symbol_cfg.order_price or price,
        signal_score=symbol_cfg.signal_score,
        position_size=symbol_cfg.position_size,
        stop_loss=symbol_cfg.stop_loss,
        take_profit=symbol_cfg.take_profit,
        market_regime=symbol_cfg.market_regime,
        model_version=symbol_cfg.model_version,
        sector=symbol_cfg.sector,
        account_equity=symbol_cfg.account_equity,
        risk_per_trade=symbol_cfg.risk_per_trade,
        cash_available=symbol_cfg.cash_available,
        expected_gross_edge_bps=symbol_cfg.expected_gross_edge_bps,
        expected_win_bps=symbol_cfg.expected_win_bps,
        expected_loss_bps=symbol_cfg.expected_loss_bps,
        expected_sharpe=symbol_cfg.expected_sharpe,
        commission_rate=symbol_cfg.commission_rate,
        sell_tax_rate=symbol_cfg.sell_tax_rate,
        spread_bps=symbol_cfg.spread_bps,
        slippage_bps=symbol_cfg.slippage_bps,
        fx_spread_bps=symbol_cfg.fx_spread_bps,
        leverage=symbol_cfg.leverage,
        margin_interest_rate=symbol_cfg.margin_interest_rate,
        borrow_fee_rate=symbol_cfg.borrow_fee_rate,
        expected_holding_days=symbol_cfg.expected_holding_days,
        fill_probability=symbol_cfg.fill_probability,
        is_short=symbol_cfg.is_short,
        market_beta=symbol_cfg.market_beta,
    )


def _auto_client_order_id(
    session_id: str | None,
    symbol: str,
    side: str,
    price: float,
    quantity: int,
) -> str | None:
    if not session_id:
        return None
    return f"auto:{session_id}:{symbol}:{side}:{int(price)}:{quantity}"


def _resolve_loop_price(symbol_cfg: AutoTradeSymbolConfig) -> dict[str, Any]:
    if symbol_cfg.price is not None:
        return {
            "price": symbol_cfg.price,
            "source": "request",
            "message": "Using request fallback price",
        }

    price_data = fetch_price_data(symbol_cfg.symbol)
    current_price = price_data.get("current_price")
    if isinstance(current_price, (int, float)) and current_price > 0:
        return {
            "price": float(current_price),
            "source": price_data.get("source", "market_data"),
            "message": "Using current market price",
        }
    return {
        "price": None,
        "source": price_data.get("source", "market_data"),
        "message": price_data.get("message") or "No usable current price; provide price",
    }


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _validate_start_request(req: AutoTradeStartRequest) -> None:
    if not req.symbols and not req.auto_discover_symbols:
        raise AutoTradingError(
            "symbols are required when auto_discover_symbols is false",
            status_code=400,
        )
    if req.execution_mode == "live":
        if not req.live_confirm_token:
            raise AutoTradingError(
                "live_confirm_token is required for live auto-trading",
                status_code=400,
            )
        if not settings.enable_live_trading:
            raise AutoTradingError("Live trading is disabled", status_code=403)
        if settings.kis_is_paper:
            raise AutoTradingError(
                "Live auto-trading requires KIS_IS_PAPER=false",
                status_code=403,
            )
        if req.live_confirm_token != settings.live_trading_confirm_token:
            raise AutoTradingError("Invalid live_confirm_token", status_code=403)


def _embedded_worker_status() -> dict[str, Any]:
    try:
        from app.workers.manager import embedded_worker_status

        return embedded_worker_status()
    except Exception as exc:
        return {"enabled": settings.embedded_workers_enabled, "error": str(exc)}


def _start_embedded_workers_if_enabled() -> dict[str, Any]:
    if not settings.embedded_workers_enabled:
        return _embedded_worker_status()
    try:
        from app.workers.manager import ensure_embedded_workers_started

        return ensure_embedded_workers_started()
    except Exception as exc:
        return {"enabled": settings.embedded_workers_enabled, "error": str(exc)}
