from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import sqlite3
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
from app.trading.edge_calibration import edge_entry_gate
from app.trading.order_approval import (
    OrderApprovalError,
    confirm_order_preview,
    create_order_preview,
)
from app.trading import auto_trading_store
from app.trading import broker_sync
from app.trading import order_state
from app.trading import paper_trading
from app.trading import risk_manager
from app.trading.atr_exits import atr_exit_levels_from_price_data
from app.trading.edge_calibration import refresh_edge_training_samples
from app.trading.universe_scanner import (
    scan_universe_for_auto_trade,
    scanner_candidate_to_symbol_config,
)


class AutoTradingError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def start_auto_trading(req: AutoTradeStartRequest) -> dict[str, Any]:
    """Persist an auto-trading session for a separate worker process."""
    _validate_start_request(req)
    account_key = auto_trading_store.account_key_for_request(req)
    if settings.auto_trading_one_session_per_account:
        existing = auto_trading_store.get_active_session_for_account(account_key)
        if existing:
            return {
                "session_id": existing["session_id"],
                "status": existing["status"],
                "account_key": existing.get("account_key"),
                "execution_mode": req.execution_mode,
                "interval_seconds": existing["interval_seconds"],
                "max_cycles": existing["max_cycles"],
                "started_at": existing["created_at"],
                "message": (
                    "An active auto-trading session already exists for this account; "
                    "no duplicate session was created."
                ),
                "universe_scan": None,
            }
    session = auto_trading_store.create_session(req)
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "account_key": session.get("account_key") or account_key,
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
    auto_trading_store.recover_overdue_active_sessions()
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
        scan = scan_universe_for_auto_trade(
            req,
            worker_id=session_id or "inline-auto-trading",
        )
        symbols = scan["symbols"]
        symbols.extend(
            _managed_position_exit_symbols(
                req=req,
                active_candidate_symbols=set(scan.get("active_candidate_symbols") or []),
            )
        )
        results_prefix.append(
            {
                "symbol": "__universe__",
                "status": scan["status"],
                "scan_id": scan["scan_id"],
                "source_symbol_count": scan["source_symbol_count"],
                "snapshot_count": scan.get("snapshot_count", scan["source_symbol_count"]),
                "candidate_count": scan["candidate_count"],
                "final_count": scan["final_count"],
                "executable_count": scan.get("executable_count"),
                "final_candidates": scan["final_candidates"],
                "ready_candidates": scan.get("ready_candidates", []),
                "worker_hurdle_rate": scan.get("worker_hurdle_rate"),
                "active_candidate_symbols": scan.get("active_candidate_symbols", []),
                "entry_gate": scan.get("entry_gate"),
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
        if settings.edge_calibration_enabled and settings.edge_calibration_refresh_after_scan:
            try:
                label_refresh = refresh_edge_training_samples()
                results_prefix[0]["label_refresh"] = label_refresh
                results_prefix[0]["stored_sample_count"] = label_refresh.get(
                    "stored_sample_count"
                )
            except Exception as exc:
                results_prefix[0]["label_refresh"] = {
                    "status": "error",
                    "message": str(exc),
                }

    if _requires_live_exit_confirmation(req) and any(
        symbol_cfg.requested_action == "exit" for symbol_cfg in symbols
    ):
        return results_prefix + _run_live_exits_then_entries(
            req=req,
            symbols=symbols,
            session_id=session_id,
        )

    prepared = _prepare_symbols_for_account_balance(req, symbols)
    results_prefix.extend(prepared["results"])
    symbols = prepared["symbols"]
    if not symbols:
        return results_prefix

    return results_prefix + _run_ordered_symbols(
        req=req,
        symbols=symbols,
        session_id=session_id,
    )


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


def run_orchestrated_candidates_once(
    req: AutoTradeStartRequest,
    *,
    active_candidates: list[dict[str, Any]],
    session_id: str | None = None,
    execute_entries: bool | None = None,
) -> dict[str, Any]:
    execute_entries = (
        settings.trade_orchestrator_execute_entries
        if execute_entries is None
        else execute_entries
    )
    plan = build_orchestrated_symbol_plan(
        req,
        active_candidates=active_candidates,
        execute_entries=bool(execute_entries),
    )
    results: list[dict[str, Any]] = []

    if plan["exit_symbols"]:
        results.extend(
            _run_symbol_batch(
                req=req,
                symbols=plan["exit_symbols"],
                session_id=session_id,
            )
        )
        if plan["entry_symbols"] and _requires_live_exit_confirmation(req):
            confirmation = _confirm_live_exits_before_entries(plan["exit_symbols"])
            results.append(confirmation)
            if confirmation["status"] != "confirmed":
                return _orchestrated_result_payload(
                    status="blocked",
                    message=(
                        "Live exits are not fully confirmed yet; entries were held back."
                    ),
                    plan=plan,
                    results=results,
                )

    if plan["entry_symbols"]:
        prepared = _prepare_symbols_for_account_balance(req, plan["entry_symbols"])
        results.extend(prepared["results"])
        results.extend(
            _run_symbol_batch(
                req=req,
                symbols=prepared["symbols"],
                session_id=session_id,
            )
        )

    status = "idle"
    if any(result.get("status") in ("confirmed", "success", "pending") for result in results):
        status = "executed"
    elif results:
        status = "blocked"
    elif plan["active_candidate_symbols"] and not plan["entry_gate"].get("approved", False):
        status = "blocked"

    return _orchestrated_result_payload(
        status=status,
        message=_orchestrated_message(plan=plan, status=status),
        plan=plan,
        results=results,
    )


def _orchestrated_message(*, plan: dict[str, Any], status: str) -> str:
    if status == "blocked" and not plan["entry_gate"].get("approved", False):
        return str(
            plan["entry_gate"].get("message")
            or "Trade orchestrator entry gate blocked entries"
        )
    return "Trade orchestrator compared scanner candidates with open positions"


def _orchestrated_result_payload(
    *,
    status: str,
    message: str,
    plan: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "active_candidate_symbols": plan["active_candidate_symbols"],
        "planned_exit_count": len(plan["exit_symbols"]),
        "planned_entry_count": len(plan["entry_symbols"]),
        "planned_exits": [
            {"symbol": item.symbol, "quantity": item.quantity}
            for item in plan["exit_symbols"]
        ],
        "planned_entries": [
            {
                "symbol": item.symbol,
                "signal_score": item.signal_score,
                "position_size": item.position_size,
                "cash_available": item.cash_available,
            }
            for item in plan["entry_symbols"]
        ],
        "entry_gate": plan["entry_gate"],
        "results": results,
    }


def build_orchestrated_symbol_plan(
    req: AutoTradeStartRequest,
    *,
    active_candidates: list[dict[str, Any]],
    execute_entries: bool = True,
) -> dict[str, Any]:
    active_candidate_symbols = {
        str(candidate.get("symbol"))
        for candidate in active_candidates
        if candidate.get("symbol")
    }
    positions = _open_positions(req.execution_mode)
    exit_symbols = _managed_position_exit_symbols(
        req=req,
        active_candidate_symbols=active_candidate_symbols,
    )
    entry_gate = _edge_entry_gate_for_mode(
        active_candidates,
        execution_mode=req.execution_mode,
    )

    entry_symbols: list[AutoTradeSymbolConfig] = []
    if execute_entries and entry_gate.get("approved", False):
        entry_symbols = _entry_symbols_from_scanner_candidates(
            req=req,
            active_candidates=active_candidates,
            open_symbols=set(positions.keys()),
        )

    return {
        "active_candidate_symbols": sorted(active_candidate_symbols),
        "open_symbols": sorted(positions.keys()),
        "exit_symbols": exit_symbols,
        "entry_symbols": entry_symbols,
        "entry_gate": entry_gate,
    }


def _edge_entry_gate_for_mode(
    candidates: list[dict[str, Any]],
    *,
    execution_mode: str,
) -> dict[str, Any]:
    try:
        return edge_entry_gate(candidates, execution_mode=execution_mode)
    except TypeError as exc:
        if "execution_mode" not in str(exc):
            raise
        return edge_entry_gate(candidates)


def _entry_symbols_from_scanner_candidates(
    *,
    req: AutoTradeStartRequest,
    active_candidates: list[dict[str, Any]],
    open_symbols: set[str],
) -> list[AutoTradeSymbolConfig]:
    now = _now()
    hurdle_rate = float(settings.universe_scanner_worker_hurdle_rate_bps or 0.0)
    rows = sorted(
        active_candidates,
        key=lambda item: (
            float(item.get("net_edge") or 0),
            float(item.get("composite_score") or 0),
            -int(item.get("rank") or 999),
        ),
        reverse=True,
    )
    symbols: list[AutoTradeSymbolConfig] = []
    for candidate in rows:
        symbol = str(candidate.get("symbol") or "")
        if not symbol or symbol in open_symbols:
            continue
        if candidate.get("status") not in ("READY", "CLAIMED"):
            continue
        if str(candidate.get("expires_at") or "") <= now:
            continue
        if float(candidate.get("net_edge") or 0.0) <= hurdle_rate:
            continue
        symbols.append(scanner_candidate_to_symbol_config(req, candidate))
    return symbols


def _run_ordered_symbols(
    *,
    req: AutoTradeStartRequest,
    symbols: list[AutoTradeSymbolConfig],
    session_id: str | None,
) -> list[dict[str, Any]]:
    exit_symbols = [
        symbol_cfg
        for symbol_cfg in symbols
        if symbol_cfg.requested_action == "exit"
    ]
    entry_symbols = [
        symbol_cfg
        for symbol_cfg in symbols
        if symbol_cfg.requested_action != "exit"
    ]
    results: list[dict[str, Any]] = []
    if exit_symbols:
        results.extend(
            _run_symbol_batch(
                req=req,
                symbols=exit_symbols,
                session_id=session_id,
            )
        )
        if entry_symbols and _requires_live_exit_confirmation(req):
            confirmation = _confirm_live_exits_before_entries(exit_symbols)
            results.append(confirmation)
            if confirmation["status"] != "confirmed":
                return results
    if entry_symbols:
        results.extend(
            _run_symbol_batch(
                req=req,
                symbols=entry_symbols,
                session_id=session_id,
            )
        )
    return results


def _run_live_exits_then_entries(
    *,
    req: AutoTradeStartRequest,
    symbols: list[AutoTradeSymbolConfig],
    session_id: str | None,
) -> list[dict[str, Any]]:
    exit_symbols = [
        symbol_cfg
        for symbol_cfg in symbols
        if symbol_cfg.requested_action == "exit"
    ]
    entry_symbols = [
        symbol_cfg
        for symbol_cfg in symbols
        if symbol_cfg.requested_action != "exit"
    ]
    results = _run_symbol_batch(
        req=req,
        symbols=exit_symbols,
        session_id=session_id,
    )
    if not entry_symbols:
        return results
    confirmation = _confirm_live_exits_before_entries(exit_symbols)
    results.append(confirmation)
    if confirmation["status"] != "confirmed":
        return results
    prepared = _prepare_symbols_for_account_balance(req, entry_symbols)
    results.extend(prepared["results"])
    results.extend(
        _run_symbol_batch(
            req=req,
            symbols=prepared["symbols"],
            session_id=session_id,
        )
    )
    return results


def _run_symbol_batch(
    *,
    req: AutoTradeStartRequest,
    symbols: list[AutoTradeSymbolConfig],
    session_id: str | None,
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    if len(symbols) <= 1:
        return [_run_symbol(req, symbols[0], session_id=session_id)]
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
    return [result for result in results if result is not None]


def _requires_live_exit_confirmation(req: AutoTradeStartRequest) -> bool:
    return bool(
        req.execution_mode == "live"
        and settings.live_exit_confirm_before_entry
    )


def _confirm_live_exits_before_entries(
    exit_symbols: list[AutoTradeSymbolConfig],
) -> dict[str, Any]:
    try:
        sync_result = broker_sync.sync_kis_account()
    except Exception as exc:
        return {
            "symbol": "__exit_confirmation__",
            "status": "blocked",
            "message": f"Broker sync failed; entries are blocked until exits are confirmed: {exc}",
        }

    if sync_result.get("status") != "success":
        return {
            "symbol": "__exit_confirmation__",
            "status": "blocked",
            "message": "Broker sync did not succeed; entries are blocked until exits are confirmed.",
            "broker_sync": sync_result,
        }

    account_no = str(sync_result.get("account_no") or settings.kis_account_no or "")
    pending: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    live_positions = _open_live_positions()
    for symbol_cfg in exit_symbols:
        state = order_state.reconcile_after_broker_sync(
            symbol=symbol_cfg.symbol,
            market=symbol_cfg.market,
            account_no=account_no,
        )
        position = live_positions.get(symbol_cfg.symbol) or {}
        observed_quantity = _to_int(state.get("current_quantity"))
        if observed_quantity is None:
            observed_quantity = _to_int(position.get("quantity")) or 0
        state_name = str(state.get("state") or "UNKNOWN")
        item = {
            "symbol": symbol_cfg.symbol,
            "state": state_name,
            "current_quantity": int(observed_quantity or 0),
            "order_status": state.get("raw", {}).get("message"),
        }
        states.append(item)
        if state_name in order_state.PENDING_POSITION_STATES or int(observed_quantity or 0) > 0:
            pending.append(item)

    if pending:
        return {
            "symbol": "__exit_confirmation__",
            "status": "blocked",
            "message": "One or more live exits are still pending or partially filled; entries were skipped.",
            "pending_exits": pending,
            "states": states,
            "broker_sync": sync_result,
        }
    return {
        "symbol": "__exit_confirmation__",
        "status": "confirmed",
        "message": "All live exits are confirmed flat; entries may proceed.",
        "states": states,
        "broker_sync": sync_result,
    }


def _managed_position_exit_symbols(
    *,
    req: AutoTradeStartRequest,
    active_candidate_symbols: set[str],
) -> list[AutoTradeSymbolConfig]:
    by_symbol: dict[str, AutoTradeSymbolConfig] = {}
    for item in _expired_candidate_exit_symbols(
        req=req,
        active_candidate_symbols=active_candidate_symbols,
    ):
        by_symbol[item.symbol] = item
    for item in _time_stop_exit_symbols(req=req):
        by_symbol[item.symbol] = item
    return list(by_symbol.values())


def _expired_candidate_exit_symbols(
    *,
    req: AutoTradeStartRequest,
    active_candidate_symbols: set[str],
) -> list[AutoTradeSymbolConfig]:
    positions = _open_positions(req.execution_mode)
    exit_symbols: list[AutoTradeSymbolConfig] = []
    for symbol, position in positions.items():
        if symbol in active_candidate_symbols:
            continue
        quantity = int(position.get("quantity") or 0)
        if quantity <= 0:
            continue
        price = _to_float(position.get("current_price"))
        exit_symbols.append(
            AutoTradeSymbolConfig(
                symbol=symbol,
                name=position.get("name"),
                market="KR",
                strategy_type="swing",
                risk_level="medium",
                requested_action="exit",
                price=price,
                decision_price=price,
                order_price=price,
                quantity=quantity,
                signal_score=0,
                expected_holding_days=5.0,
            )
        )
    return exit_symbols


def _time_stop_exit_symbols(
    *,
    req: AutoTradeStartRequest,
) -> list[AutoTradeSymbolConfig]:
    max_days = max(0, int(settings.position_time_stop_trading_days or 0))
    if max_days <= 0:
        return []
    positions = _open_positions(req.execution_mode)
    exit_symbols: list[AutoTradeSymbolConfig] = []
    now = datetime.now()
    for symbol, position in positions.items():
        quantity = int(position.get("quantity") or 0)
        if quantity <= 0:
            continue
        opened_at = _position_opened_at(position)
        if opened_at is None:
            continue
        elapsed = _trading_days_elapsed(opened_at, now)
        if elapsed < max_days:
            continue
        price = _to_float(position.get("current_price"))
        exit_symbols.append(
            AutoTradeSymbolConfig(
                symbol=symbol,
                name=position.get("name"),
                market="KR",
                strategy_type="swing",
                risk_level="medium",
                requested_action="exit",
                price=price,
                decision_price=price,
                order_price=price,
                quantity=quantity,
                signal_score=0,
                expected_holding_days=float(max_days),
            )
        )
    return exit_symbols


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
                price_result.get("price_data") or {},
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
    max_open_positions = max(1, int(settings.auto_trading_max_open_positions or 5))
    open_symbols = set(_open_positions(str(account.get("mode") or "paper")).keys())
    projected_open_count = len(open_symbols)
    blocked_by_index: dict[int, dict[str, Any]] = {}
    allocated_by_index: dict[int, AutoTradeSymbolConfig] = {}
    order = sorted(
        range(len(symbols)),
        key=lambda index: (
            symbols[index].requested_action == "exit",
            float(symbols[index].signal_score or 0),
        ),
        reverse=True,
    )

    for index in order:
        symbol_cfg = symbols[index]
        if symbol_cfg.requested_action == "exit":
            allocated_by_index[index] = symbol_cfg
            open_symbols.discard(symbol_cfg.symbol)
            projected_open_count = max(0, projected_open_count - 1)
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

        opens_new_position = symbol_cfg.symbol not in open_symbols
        if opens_new_position and projected_open_count >= max_open_positions:
            blocked_by_index[index] = {
                "symbol": symbol_cfg.symbol,
                "status": "blocked",
                "message": (
                    "Maximum open-position count reached; no order preview or "
                    "broker order was attempted."
                ),
                "max_open_positions": max_open_positions,
                "open_position_count": projected_open_count,
                "open_symbols": sorted(open_symbols),
            }
            continue

        circuit_breaker = _strategy_circuit_breaker_for_symbol(symbol_cfg, account)
        if circuit_breaker.get("action") == "pause":
            blocked_by_index[index] = {
                "symbol": symbol_cfg.symbol,
                "status": "blocked",
                "message": str(
                    circuit_breaker.get("message")
                    or "Strategy circuit breaker paused entries"
                ),
                "strategy_circuit_breaker": circuit_breaker,
            }
            continue
        target_weight = _target_position_weight(symbol_cfg) * float(
            circuit_breaker.get("position_scale") or 1.0
        )
        target_weight = max(0.0, target_weight)
        max_allocation = min(remaining_cash, account_equity * target_weight)
        if max_allocation <= 0:
            blocked_by_index[index] = _insufficient_cash_result(
                symbol_cfg=symbol_cfg,
                available_cash=remaining_cash,
                account=account,
                reason="position sizing allocation is zero",
            )
            continue

        requested_quantity = symbol_cfg.quantity
        if requested_quantity is not None:
            requested_quantity = min(
                int(requested_quantity),
                int(max_allocation / price),
            )
            if requested_quantity <= 0:
                blocked_by_index[index] = _insufficient_cash_result(
                    symbol_cfg=symbol_cfg,
                    available_cash=remaining_cash,
                    account=account,
                    reason="position cap is below the minimum executable quantity",
                )
                continue
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
            symbol_cfg = symbol_cfg.model_copy(update={"quantity": requested_quantity})
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
                cash_available=max_allocation,
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
            symbol_cfg = symbol_cfg.model_copy(update={"quantity": recommended_quantity})

        remaining_cash = max(0.0, remaining_cash - allocation)
        if opens_new_position:
            open_symbols.add(symbol_cfg.symbol)
            projected_open_count += 1
        allocated_by_index[index] = symbol_cfg.model_copy(
            update={
                "account_equity": account_equity,
                "cash_available": allocation,
                "position_size": round(allocation / account_equity, 6)
                if account_equity
                else None,
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


def _target_position_weight(symbol_cfg: AutoTradeSymbolConfig) -> float:
    base = max(0.0, float(settings.trade_orchestrator_base_position_weight or 0.10))
    min_weight = max(0.0, float(settings.trade_orchestrator_min_position_weight or 0.0))
    max_weight = max(min_weight, float(settings.trade_orchestrator_max_position_weight or base))
    edge_cap = max(1.0, float(settings.trade_orchestrator_edge_weight_cap_bps or 300.0))
    edge = _to_float(symbol_cfg.expected_gross_edge_bps)
    if edge is None:
        edge = _to_float(symbol_cfg.signal_score)
        edge = (edge or 50.0) / 100.0 * edge_cap
    edge_ratio = max(0.0, min(1.0, float(edge) / edge_cap))
    weighted = base * (0.75 + edge_ratio * 0.75)
    return round(max(min_weight, min(max_weight, weighted)), 6)


def _strategy_circuit_breaker_for_symbol(
    symbol_cfg: AutoTradeSymbolConfig,
    account: dict[str, Any],
) -> dict[str, Any]:
    path = settings.storage_path(paper_trading.DEFAULT_DB_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            paper_trading.initialize_db(conn)
            return risk_manager.strategy_circuit_breaker(
                conn=conn,
                market=symbol_cfg.market,
                strategy_type=symbol_cfg.strategy_type,
            )
    except Exception as exc:
        return {
            "enabled": bool(settings.strategy_circuit_breaker_enabled),
            "action": "none",
            "position_scale": 1.0,
            "message": f"Strategy circuit breaker unavailable: {exc}",
            "mode": account.get("mode"),
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
        symbol_cfg = _apply_order_sizing_defaults(
            req,
            symbol_cfg,
            price,
            price_result.get("price_data") or {},
        )
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
        trailing_stop=symbol_cfg.trailing_stop,
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
    price_data: dict[str, Any] | None = None,
) -> AutoTradeSymbolConfig:
    updates: dict[str, Any] = {}
    if symbol_cfg.account_equity is None:
        updates["account_equity"] = req.account_equity
    if symbol_cfg.risk_per_trade is None:
        updates["risk_per_trade"] = req.risk_per_trade
    if symbol_cfg.cash_available is None and req.cash_available is not None:
        updates["cash_available"] = req.cash_available
    if (
        symbol_cfg.quantity is None
        and (
            symbol_cfg.stop_loss is None
            or symbol_cfg.take_profit is None
            or symbol_cfg.trailing_stop is None
        )
    ):
        levels = atr_exit_levels_from_price_data(
            entry_price=price,
            price_data=price_data or {},
        )
        if symbol_cfg.stop_loss is None and levels["stop_loss"] is not None:
            updates["stop_loss"] = levels["stop_loss"]
        if symbol_cfg.take_profit is None and levels["take_profit"] is not None:
            updates["take_profit"] = levels["take_profit"]
        if symbol_cfg.trailing_stop is None and levels["trailing_stop"] is not None:
            updates["trailing_stop"] = levels["trailing_stop"]
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
        strategy_type=symbol_cfg.strategy_type,
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
        trailing_stop=symbol_cfg.trailing_stop,
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
        price_data: dict[str, Any] = {}
        if (
            symbol_cfg.quantity is None
            and (
                symbol_cfg.stop_loss is None
                or symbol_cfg.take_profit is None
                or symbol_cfg.trailing_stop is None
            )
        ):
            try:
                price_data = fetch_price_data(symbol_cfg.symbol)
            except Exception:
                price_data = {}
        return {
            "price": symbol_cfg.price,
            "source": "request",
            "message": "Using request fallback price",
            "price_data": price_data,
        }

    price_data = fetch_price_data(symbol_cfg.symbol)
    current_price = price_data.get("current_price")
    if isinstance(current_price, (int, float)) and current_price > 0:
        return {
            "price": float(current_price),
            "source": price_data.get("source", "market_data"),
            "message": "Using current market price",
            "price_data": price_data,
        }
    return {
        "price": None,
        "source": price_data.get("source", "market_data"),
        "message": price_data.get("message") or "No usable current price; provide price",
    }


def _open_positions(mode: str) -> dict[str, dict[str, Any]]:
    if mode == "live":
        return _open_live_positions()
    return _open_paper_positions()


def _open_paper_positions() -> dict[str, dict[str, Any]]:
    path = settings.storage_path(paper_trading.DEFAULT_DB_PATH)
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            paper_trading.initialize_db(conn)
            rows = conn.execute(
                """
                SELECT symbol, name, quantity, avg_price AS current_price,
                       opened_at, updated_at
                FROM positions
                WHERE quantity > 0
                """
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {str(row["symbol"]): dict(row) for row in rows}


def _open_live_positions() -> dict[str, dict[str, Any]]:
    path = settings.storage_path(settings.broker_sync_db_path)
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(broker_sync.SCHEMA_SQL)
            broker_sync._ensure_column(conn, "broker_positions", "opened_at", "TEXT")
            rows = conn.execute(
                """
                SELECT symbol, name, quantity, current_price, opened_at, synced_at
                FROM broker_positions
                WHERE broker = 'KIS'
                  AND quantity > 0
                """
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {str(row["symbol"]): dict(row) for row in rows}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _position_opened_at(position: dict[str, Any]) -> datetime | None:
    for key in ("opened_at", "updated_at", "synced_at"):
        raw = position.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            continue
    return None


def _trading_days_elapsed(start: datetime, end: datetime) -> int:
    if end.date() <= start.date():
        return 0
    days = 0
    current = start.date()
    while current < end.date():
        current = current + timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
