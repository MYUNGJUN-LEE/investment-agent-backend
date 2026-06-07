from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import math
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
from app.trading.live_trading import (
    LiveTradingError,
    broker_paper_safety_check,
    execute_broker_paper_order,
    execute_live_order,
)
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
from app.trading.execution_status import print_startup_log
from app.trading.fill_quality import (
    fill_quality_adjustment_for_candidate,
    record_fill_quality_event,
)
from app.trading.outcome_attribution import record_outcome_attribution
from app.trading.regime_gate import regime_gate_for_mode
from app.trading.universe_scanner import (
    get_latest_universe_scan,
    scan_universe_for_auto_trade,
    scanner_candidate_to_symbol_config,
)
from app.storage.sqlite import (
    RecoverableSQLiteError,
    connect_sqlite,
    is_recoverable_sqlite_error,
)


class AutoTradingError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def start_auto_trading(req: AutoTradeStartRequest) -> dict[str, Any]:
    """Persist an auto-trading session for a separate worker process."""
    _validate_start_request(req)
    account_key = auto_trading_store.account_key_for_request(req)
    recovery_applied = False
    if settings.auto_trading_one_session_per_account:
        recovered = auto_trading_store.recover_overdue_active_sessions()
        recovery_applied = bool(recovered)
        existing = auto_trading_store.get_active_session_for_account(account_key)
        if existing:
            return {
                "session_id": existing["session_id"],
                "status": existing["status"],
                "account_key": existing.get("account_key"),
                "execution_mode": req.execution_mode,
                "broker_provider": req.broker_provider,
                "interval_seconds": existing["interval_seconds"],
                "max_cycles": existing["max_cycles"],
                "started_at": existing["created_at"],
                "auto_recovery_applied": bool(
                    recovery_applied or existing.get("recovery_applied")
                ),
                "message": (
                    "An active auto-trading session already exists for this account; "
                    "no duplicate session was created."
                ),
                "universe_scan": None,
            }
        recovery_applied = recovery_applied or _recover_previous_locked_error_session(account_key)
    session = auto_trading_store.create_session(req)
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "account_key": session.get("account_key") or account_key,
        "execution_mode": req.execution_mode,
        "broker_provider": req.broker_provider,
        "interval_seconds": req.interval_seconds,
        "max_cycles": req.max_cycles,
        "started_at": session["created_at"],
        "auto_recovery_applied": bool(
            recovery_applied or session.get("recovery_applied")
        ),
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
    latest = sessions[0] if sessions else None
    return {
        "count": len(sessions),
        "active_session_count": (
            len(sessions)
            if status == "active"
            else len(auto_trading_store.list_sessions(status="active", limit=500))
        ),
        "latest_session_status": latest.get("status") if latest else None,
        "last_recoverable_error": latest.get("last_recoverable_error") if latest else None,
        "last_cycle_error": latest.get("last_cycle_error") if latest else None,
        "auto_recovery_applied": bool(latest.get("recovery_applied")) if latest else False,
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
    recent_sessions = list_auto_trading_sessions(limit=10)["sessions"]
    latest = recent_sessions[0] if recent_sessions else None
    if req.command == "status":
        return {
            "status": "success",
            "command": req.command,
            "message": f"{len(active)} active auto-trading session(s)",
            "active_session_count": len(active),
            "latest_session_status": latest.get("status") if latest else None,
            "last_recoverable_error": latest.get("last_recoverable_error") if latest else None,
            "last_cycle_error": latest.get("last_cycle_error") if latest else None,
            "auto_recovery_applied": bool(latest.get("recovery_applied")) if latest else False,
            "active_sessions": active,
            "recent_sessions": recent_sessions,
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
            "active_session_count": 0,
            "latest_session_status": latest.get("status") if latest else None,
            "last_recoverable_error": latest.get("last_recoverable_error") if latest else None,
            "last_cycle_error": latest.get("last_cycle_error") if latest else None,
            "auto_recovery_applied": bool(latest.get("recovery_applied")) if latest else False,
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
            "active_session_count": len(active),
            "latest_session_status": latest.get("status") if latest else None,
            "last_recoverable_error": latest.get("last_recoverable_error") if latest else None,
            "last_cycle_error": latest.get("last_cycle_error") if latest else None,
            "auto_recovery_applied": bool(latest.get("recovery_applied")) if latest else False,
            "active_sessions": active,
            "recent_sessions": recent_sessions,
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
        broker_provider=req.broker_provider,
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
        "active_session_count": len(list_auto_trading_sessions(status="active", limit=10)["sessions"]),
        "latest_session_status": started.get("status"),
        "last_recoverable_error": None,
        "last_cycle_error": None,
        "auto_recovery_applied": bool(started.get("auto_recovery_applied")),
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
            updated = auto_trading_store.complete_cycle(
                session["session_id"],
                results,
                worker_id=worker_id,
            )
            processed.append(
                {
                    "session_id": session["session_id"],
                    "status": (updated or {}).get("status", "unknown"),
                    "results": results,
                }
            )
        except Exception as exc:
            if _is_recoverable_runtime_error(exc):
                try:
                    updated = auto_trading_store.recoverable_cycle_error(
                        session["session_id"],
                        str(exc),
                        worker_id=worker_id,
                    )
                except Exception as recovery_exc:
                    processed.append(
                        {
                            "session_id": session["session_id"],
                            "status": "recoverable_error_not_recorded",
                            "recoverable_error": str(exc),
                            "recovery_write_error": str(recovery_exc),
                        }
                    )
                    continue
                processed.append(
                    {
                        "session_id": session["session_id"],
                        "status": (updated or {}).get("status", "active"),
                        "recoverable_error": str(exc),
                    }
                )
                continue

            try:
                updated = auto_trading_store.fail_cycle(
                    session["session_id"],
                    str(exc),
                    worker_id=worker_id,
                )
            except Exception as failure_write_exc:
                processed.append(
                    {
                        "session_id": session["session_id"],
                        "status": "error_not_recorded",
                        "error": str(exc),
                        "failure_write_error": str(failure_write_exc),
                    }
                )
                continue
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
    order_state.initialize_order_state_db()
    print_startup_log(
        auto_trading_worker_enabled=True,
        scanner_worker_enabled=bool(settings.universe_full_scan_enabled),
    )
    while True:
        process_due_sessions(worker_id=worker_id)
        time.sleep(float(poll_seconds))


def _is_recoverable_runtime_error(exc: BaseException) -> bool:
    return isinstance(exc, RecoverableSQLiteError) or is_recoverable_sqlite_error(exc)


def _recover_previous_locked_error_session(account_key: str) -> bool:
    recent = auto_trading_store.list_sessions(limit=20)
    return any(
        session.get("account_key") == account_key
        and session.get("status") == "error"
        and is_recoverable_sqlite_error(session.get("last_error"))
        for session in recent
    )


def _scanner_staleness_report() -> dict[str, Any]:
    latest = get_latest_universe_scan()
    now = datetime.now()
    latest_time = (
        _parse_iso_datetime(latest.get("created_at"))
        or _parse_iso_datetime(latest.get("scan_time"))
    )
    latest_age = None
    if latest_time is not None:
        now_for_age = datetime.now(latest_time.tzinfo) if latest_time.tzinfo else now
        latest_age = max(0, int((now_for_age - latest_time).total_seconds()))

    stale_after = _scanner_stale_after_seconds_for_cycle()
    is_stale = bool(
        latest.get("status") not in {"empty", "not_found"}
        and latest_age is not None
        and latest_age > stale_after
    )

    return {
        "scanner_is_stale": is_stale,
        "latest_scan_age_seconds": latest_age,
        "scanner_stale_after_seconds": stale_after,
    }


def _scanner_stale_after_seconds_for_cycle() -> int:
    source_count = max(1, int(settings.universe_scanner_max_source_symbols or 1))
    symbol_interval = max(
        0.0,
        float(settings.universe_scanner_symbol_interval_seconds or 0.0),
    )
    cap = max(0.0, float(settings.universe_scanner_symbol_interval_cap_seconds or 0.0))
    if cap > 0:
        symbol_interval = min(symbol_interval, cap)
    estimated_scan_seconds = int(source_count * symbol_interval) + 300
    return max(
        900,
        int(settings.universe_scanner_min_interval_seconds or 0),
        estimated_scan_seconds * 2,
    )


def _scan_market_coverage_fields(scan: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_source_symbol_count",
        "kospi_source_symbol_count",
        "kosdaq_source_symbol_count",
        "konex_source_symbol_count",
        "unknown_market_symbol_count",
        "kospi_snapshot_count",
        "kosdaq_snapshot_count",
        "konex_snapshot_count",
        "unknown_market_snapshot_count",
        "kospi_candidate_count",
        "kosdaq_candidate_count",
        "konex_candidate_count",
        "unknown_market_candidate_count",
        "kospi_final_candidate_count",
        "kosdaq_final_candidate_count",
        "konex_final_candidate_count",
        "unknown_market_final_candidate_count",
        "market_coverage",
    )
    return {key: scan[key] for key in keys if key in scan}


def _run_cycle(
    req: AutoTradeStartRequest,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    symbols = list(req.symbols)
    results_prefix: list[dict[str, Any]] = []
    if req.auto_discover_symbols and not symbols:
        stale_check = _scanner_staleness_report()
        try:
            scan = scan_universe_for_auto_trade(
                req,
                worker_id=session_id or "inline-auto-trading",
            )
        except Exception as exc:
            return [
                {
                    "symbol": "__universe__",
                    "status": "skipped",
                    "message": f"Universe scanner recovery failed: {exc}",
                    "scanner_is_stale": stale_check["scanner_is_stale"],
                    "latest_scan_age_seconds": stale_check["latest_scan_age_seconds"],
                    "scanner_stale_after_seconds": stale_check["scanner_stale_after_seconds"],
                    "last_scanner_recovery_attempt_at": datetime.now().isoformat(timespec="seconds"),
                    "last_scanner_recovery_status": "error",
                    "last_scanner_recovery_error": str(exc),
                }
            ]
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
                **_scan_market_coverage_fields(scan),
                "executable_count": scan.get("executable_count"),
                "final_candidates": scan["final_candidates"],
                "ready_candidates": scan.get("ready_candidates", []),
                "worker_hurdle_rate": scan.get("worker_hurdle_rate"),
                "active_candidate_symbols": scan.get("active_candidate_symbols", []),
                "entry_gate": scan.get("entry_gate"),
                "scanner_is_stale": stale_check["scanner_is_stale"],
                "latest_scan_age_seconds": stale_check["latest_scan_age_seconds"],
                "scanner_stale_after_seconds": stale_check["scanner_stale_after_seconds"],
                "last_scanner_recovery_attempt_at": (
                    datetime.now().isoformat(timespec="seconds")
                    if stale_check["scanner_is_stale"]
                    else None
                ),
                "last_scanner_recovery_status": (
                    "success" if stale_check["scanner_is_stale"] else "not_needed"
                ),
                "last_scanner_recovery_error": None,
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
    if req.execution_mode not in ("live", "broker_paper"):
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
                "fill_quality": item.fill_quality,
                "regime_gate": item.regime_gate,
                "signal_decay": item.signal_decay,
                "final_entry_edge": item.final_entry_edge,
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

def _scanner_hurdle_rate_for_mode(execution_mode: str) -> float:
    configured = float(settings.universe_scanner_worker_hurdle_rate_bps or 0.0)

    if execution_mode == "paper":
        if configured > 0:
            return -20.0
        return max(-20.0, configured)

    return max(0.0, configured)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    try:
        raw = str(value).strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _candidate_age_seconds(candidate: dict[str, Any]) -> float | None:
    for key in ("scan_time", "created_at", "observed_at", "updated_at"):
        dt = _parse_iso_datetime(candidate.get(key))
        if dt is None:
            continue
        now = datetime.now(dt.tzinfo) if dt.tzinfo is not None else datetime.now()
        return max(0.0, (now - dt).total_seconds())
    return None


def _final_edge_before_signal_decay(candidate: dict[str, Any]) -> float:
    for key in (
        "fill_quality_adjusted_edge",
        "portfolio_adjusted_net_edge",
        "net_edge",
    ):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _signal_decay_for_candidate(
    candidate: dict[str, Any],
    *,
    execution_mode: str,
) -> dict[str, Any]:
    base_edge = _final_edge_before_signal_decay(candidate)

    if not bool(settings.signal_decay_enabled):
        return {
            "enabled": False,
            "approved": True,
            "candidate_age_seconds": None,
            "signal_decay_multiplier": 1.0,
            "pre_signal_decay_edge": round(base_edge, 4),
            "signal_decay_adjusted_edge": round(base_edge, 4),
            "signal_decay_penalty_bps": 0.0,
            "message": "signal decay disabled",
        }

    age_seconds = _candidate_age_seconds(candidate)
    paper = str(execution_mode or "paper").lower() == "paper"

    if age_seconds is None:
        penalty = (
            0.0
            if paper
            else _safe_float(settings.signal_decay_missing_age_live_penalty_bps, 10.0)
        )
        adjusted = base_edge - penalty
        return {
            "enabled": True,
            "approved": True,
            "status": "missing_age",
            "candidate_age_seconds": None,
            "signal_decay_multiplier": 1.0,
            "pre_signal_decay_edge": round(base_edge, 4),
            "signal_decay_adjusted_edge": round(adjusted, 4),
            "signal_decay_penalty_bps": round(penalty, 4),
            "message": "candidate age missing; neutral in paper, mild penalty in live",
        }

    max_age = (
        _safe_float(settings.signal_decay_paper_max_candidate_age_seconds, 7200.0)
        if paper
        else _safe_float(settings.signal_decay_live_max_candidate_age_seconds, 3600.0)
    )

    approved = True
    message = "signal age acceptable"

    if age_seconds > max_age:
        if paper:
            message = (
                f"candidate age {age_seconds:.0f}s exceeds paper max age "
                f"{max_age:.0f}s; decay only"
            )
        else:
            approved = False
            message = (
                f"candidate age {age_seconds:.0f}s exceeds live max age "
                f"{max_age:.0f}s"
            )

    half_life = max(
        60.0,
        _safe_float(settings.signal_decay_half_life_seconds, 1800.0),
    )
    min_multiplier = _clamp_float(
        _safe_float(settings.signal_decay_min_multiplier, 0.35),
        0.0,
        1.0,
    )

    multiplier = math.exp(-age_seconds / half_life)
    multiplier = max(min_multiplier, min(1.0, multiplier))

    adjusted_edge = base_edge * multiplier
    penalty = base_edge - adjusted_edge

    return {
        "enabled": True,
        "approved": approved,
        "status": "ready",
        "candidate_age_seconds": round(age_seconds, 3),
        "max_candidate_age_seconds": round(max_age, 3),
        "signal_half_life_seconds": round(half_life, 3),
        "signal_decay_multiplier": round(multiplier, 6),
        "pre_signal_decay_edge": round(base_edge, 4),
        "signal_decay_adjusted_edge": round(adjusted_edge, 4),
        "signal_decay_penalty_bps": round(penalty, 4),
        "message": message,
    }


def _recent_symbol_returns_from_universe_db(
    symbol: str,
    *,
    lookback_prices: int | None = None,
) -> list[float]:
    """Load recent returns from universe_price_snapshots. No external API calls."""
    if not symbol:
        return []

    lookback_prices = max(
        6,
        int(lookback_prices or settings.portfolio_corr_lookback_prices or 30),
    )
    path = settings.storage_path(settings.universe_scanner_db_path)

    if not path.exists():
        return []

    try:
        conn = connect_sqlite(path)
        try:
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
                (symbol, lookback_prices),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return []

    prices: list[float] = []
    for row in rows:
        if not row or not row[0]:
            continue
        price = _safe_float(row[0])
        if price > 0:
            prices.append(price)
    prices = list(reversed(prices))

    if len(prices) < 2:
        return []

    returns: list[float] = []
    for prev, cur in zip(prices, prices[1:]):
        if prev and prev > 0:
            returns.append((cur / prev) - 1.0)

    return returns


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    min_len = min(len(xs), len(ys))
    min_required = max(3, int(settings.portfolio_corr_min_returns or 5))

    if min_len < min_required:
        return None

    xs = xs[-min_len:]
    ys = ys[-min_len:]

    mean_x = sum(xs) / min_len
    mean_y = sum(ys) / min_len

    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]

    var_x = sum(x * x for x in dx)
    var_y = sum(y * y for y in dy)

    if var_x <= 0 or var_y <= 0:
        return None

    cov = sum(x * y for x, y in zip(dx, dy))
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _correlation_penalty_bps(max_corr: float | None) -> float:
    if max_corr is None:
        return 0.0

    threshold = float(settings.portfolio_corr_threshold or 0.65)
    cap = float(settings.portfolio_corr_penalty_cap_bps or 35.0)

    if max_corr <= threshold:
        return 0.0

    return cap * min(
        1.0,
        (max_corr - threshold) / max(0.0001, 1.0 - threshold),
    )


def _candidate_sector_key(candidate: dict[str, Any]) -> str | None:
    raw = (
        candidate.get("sector")
        or candidate.get("industry")
        or candidate.get("universe_profile")
        or candidate.get("market_segment")
    )
    if not raw:
        return None
    return str(raw).strip().lower() or None


def _open_position_sector_keys(execution_mode: str) -> list[str]:
    sectors: list[str] = []

    try:
        positions = _open_positions(execution_mode)
    except Exception:
        return sectors

    for position in positions.values():
        raw = (
            position.get("sector")
            or position.get("industry")
            or position.get("universe_profile")
            or position.get("market_segment")
            or position.get("market")
        )
        if raw:
            sectors.append(str(raw).strip().lower())

    return sectors


def _sector_penalty_bps(
    candidate: dict[str, Any],
    *,
    execution_mode: str,
) -> float:
    candidate_sector = _candidate_sector_key(candidate)
    if not candidate_sector:
        return 0.0

    open_sectors = _open_position_sector_keys(execution_mode)
    same_count = sum(1 for sector in open_sectors if sector == candidate_sector)

    per_position = float(settings.portfolio_sector_penalty_bps or 10.0)
    cap = float(settings.portfolio_sector_penalty_cap_bps or 30.0)

    return min(cap, same_count * per_position)


def _neutral_portfolio_penalty(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "max_corr_with_open_positions": None,
        "correlation_penalty_bps": 0.0,
        "sector_penalty_bps": 0.0,
        "total_penalty_bps": 0.0,
    }


def _portfolio_penalty_for_candidate(
    *,
    candidate: dict[str, Any],
    open_symbols: set[str],
    execution_mode: str,
) -> dict[str, Any]:
    if not bool(settings.portfolio_penalty_enabled):
        return _neutral_portfolio_penalty(enabled=False)

    candidate_symbol = str(candidate.get("symbol") or "")
    candidate_returns = _recent_symbol_returns_from_universe_db(candidate_symbol)

    correlations: list[float] = []
    if candidate_returns:
        for open_symbol in open_symbols:
            open_returns = _recent_symbol_returns_from_universe_db(str(open_symbol))
            corr = _pearson_corr(candidate_returns, open_returns)
            if corr is not None:
                correlations.append(corr)

    max_corr = max(correlations) if correlations else None
    corr_penalty = _correlation_penalty_bps(max_corr)
    sector_penalty = _sector_penalty_bps(candidate, execution_mode=execution_mode)

    total = corr_penalty + sector_penalty

    return {
        "enabled": True,
        "max_corr_with_open_positions": round(max_corr, 6) if max_corr is not None else None,
        "correlation_penalty_bps": round(corr_penalty, 4),
        "sector_penalty_bps": round(sector_penalty, 4),
        "total_penalty_bps": round(total, 4),
    }


def _orderbook_feature_summary(
    orderbook: dict[str, Any],
    *,
    order_value: float | None = None,
) -> dict[str, Any]:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []

    if not bids or not asks:
        return {
            "status": "missing",
            "spread_bps": None,
            "order_book_imbalance": None,
            "weighted_order_book_imbalance": None,
            "microprice_edge_bps": None,
            "depth_coverage": None,
            "approved": True,
            "message": "Orderbook data unavailable; neutral pass",
        }

    top_n = max(1, min(10, int(settings.pretrade_orderbook_top_n or 3)))

    try:
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        bid_sizes = [float(item.get("size") or 0.0) for item in bids[:top_n]]
        ask_sizes = [float(item.get("size") or 0.0) for item in asks[:top_n]]
    except (TypeError, ValueError, KeyError):
        return {
            "status": "invalid",
            "spread_bps": None,
            "order_book_imbalance": None,
            "weighted_order_book_imbalance": None,
            "microprice_edge_bps": None,
            "depth_coverage": None,
            "approved": True,
            "message": "Orderbook data invalid; neutral pass",
        }

    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return {
            "status": "invalid",
            "spread_bps": None,
            "order_book_imbalance": None,
            "weighted_order_book_imbalance": None,
            "microprice_edge_bps": None,
            "depth_coverage": None,
            "approved": True,
            "message": "Orderbook quote invalid; neutral pass",
        }

    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000.0

    bid_sum = sum(bid_sizes)
    ask_sum = sum(ask_sizes)
    denom = bid_sum + ask_sum

    imbalance = None
    if denom > 0:
        imbalance = (bid_sum - ask_sum) / denom

    weighted_bid = sum(size / idx for idx, size in enumerate(bid_sizes, start=1))
    weighted_ask = sum(size / idx for idx, size in enumerate(ask_sizes, start=1))
    weighted_denom = weighted_bid + weighted_ask

    weighted_imbalance = None
    if weighted_denom > 0:
        weighted_imbalance = (weighted_bid - weighted_ask) / weighted_denom

    microprice_edge_bps = None
    if bid_sizes and ask_sizes and (bid_sizes[0] + ask_sizes[0]) > 0:
        microprice = (
            best_ask * bid_sizes[0] + best_bid * ask_sizes[0]
        ) / (bid_sizes[0] + ask_sizes[0])
        microprice_edge_bps = (microprice - mid) / mid * 10_000.0

    ask_depth_value = 0.0
    for item in asks[:top_n]:
        try:
            ask_depth_value += float(item.get("price") or 0.0) * float(
                item.get("size") or 0.0
            )
        except (TypeError, ValueError):
            continue

    depth_coverage = None
    if order_value and order_value > 0:
        depth_coverage = ask_depth_value / order_value

    approved = True
    reasons: list[str] = []

    if spread_bps > float(settings.pretrade_orderbook_max_spread_bps or 25.0):
        approved = False
        reasons.append(f"spread {spread_bps:.2f}bps too wide")

    min_imbalance = float(settings.pretrade_orderbook_min_imbalance or -0.20)
    if weighted_imbalance is not None and weighted_imbalance < min_imbalance:
        approved = False
        reasons.append(
            f"weighted imbalance {weighted_imbalance:.3f} below {min_imbalance:.3f}"
        )

    min_depth = float(settings.pretrade_orderbook_min_depth_coverage or 2.0)
    if depth_coverage is not None and depth_coverage < min_depth:
        approved = False
        reasons.append(f"depth coverage {depth_coverage:.2f} below {min_depth:.2f}")

    return {
        "status": "ready",
        "spread_bps": round(spread_bps, 4),
        "order_book_imbalance": round(imbalance, 6) if imbalance is not None else None,
        "weighted_order_book_imbalance": round(weighted_imbalance, 6)
        if weighted_imbalance is not None
        else None,
        "microprice_edge_bps": round(microprice_edge_bps, 4)
        if microprice_edge_bps is not None
        else None,
        "depth_coverage": round(depth_coverage, 4)
        if depth_coverage is not None
        else None,
        "approved": approved,
        "message": "; ".join(reasons) if reasons else "Orderbook check passed",
    }


def _fetch_orderbook_neutral(symbol: str) -> dict[str, Any]:
    """
    Try to fetch orderbook data if a KIS helper exists.
    If unavailable, return empty dict.
    This function must never raise.
    """
    try:
        from app.data_sources import kis as kis_source

        fetcher = getattr(kis_source, "fetch_orderbook_data", None)
        if fetcher is None:
            fetcher = getattr(kis_source, "fetch_order_book_data", None)

        if fetcher is None:
            return {}

        data = fetcher(symbol)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _candidate_edge_for_sizing(candidate: dict[str, Any]) -> float:
    raw = (
        candidate.get("portfolio_adjusted_net_edge")
        if candidate.get("portfolio_adjusted_net_edge") is not None
        else candidate.get("net_edge")
    )
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_has_edge_for_sizing(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("portfolio_adjusted_net_edge") is not None
        or candidate.get("net_edge") is not None
    )


def _candidate_stop_risk_bps(candidate: dict[str, Any]) -> float:
    for key in (
        "net_stop_loss_bps",
        "stop_loss_bps",
        "expected_risk",
        "expected_risk_bps",
    ):
        try:
            value = candidate.get(key)
            if value is not None and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue

    reward_risk = candidate.get("edge_reward_risk") or {}
    try:
        value = reward_risk.get("loss_risk_floor_bps")
        if value is not None and float(value) > 0:
            return float(value)
    except (AttributeError, TypeError, ValueError):
        pass

    return float(settings.position_sizing_default_stop_bps or 250.0)


def _edge_position_multiplier(candidate: dict[str, Any]) -> float:
    if not bool(settings.position_sizing_edge_enabled):
        return 1.0

    edge = _candidate_edge_for_sizing(candidate)

    edge_floor = float(settings.position_sizing_edge_floor_bps or 0.0)
    edge_cap = max(
        edge_floor + 1.0,
        float(settings.position_sizing_edge_cap_bps or 150.0),
    )

    min_multiplier = float(settings.position_sizing_min_multiplier or 0.35)
    max_multiplier = float(settings.position_sizing_max_multiplier or 1.0)

    if edge <= edge_floor:
        return min_multiplier

    raw = edge / edge_cap
    return _clamp_float(raw, min_multiplier, max_multiplier)


def _position_value_from_edge_and_risk(
    *,
    candidate: dict[str, Any],
    account_equity: float,
    risk_per_trade: float,
    regime_position_multiplier: float = 1.0,
) -> dict[str, Any]:
    equity = max(0.0, float(account_equity or 0.0))
    risk_fraction = max(0.0001, float(risk_per_trade or 0.005))

    stop_bps = max(1.0, _candidate_stop_risk_bps(candidate))
    stop_risk_pct = stop_bps / 10_000.0

    risk_budget = equity * risk_fraction
    raw_position_value = risk_budget / max(0.0001, stop_risk_pct)

    edge_multiplier = _edge_position_multiplier(candidate)

    max_symbol_weight = max(
        0.001,
        float(settings.position_sizing_max_symbol_weight or 0.10),
    )
    max_symbol_value = equity * max_symbol_weight

    final_position_value = min(
        raw_position_value * edge_multiplier,
        max_symbol_value,
    )
    regime_multiplier = _clamp_float(regime_position_multiplier, 0.0, 1.25)
    final_position_value = final_position_value * regime_multiplier

    return {
        "position_value": round(max(0.0, final_position_value), 4),
        "risk_budget": round(risk_budget, 4),
        "stop_risk_bps": round(stop_bps, 4),
        "edge_for_sizing_bps": round(_candidate_edge_for_sizing(candidate), 4),
        "edge_position_multiplier": round(edge_multiplier, 4),
        "regime_position_multiplier": round(regime_multiplier, 4),
        "max_symbol_value": round(max_symbol_value, 4),
        "position_sizing_formula": (
            "position_value = min((account_equity * risk_per_trade / stop_risk_pct) "
            "* edge_multiplier, account_equity * max_symbol_weight) * regime_position_multiplier"
        ),
    }


def _candidate_with_symbol_stop_risk(
    candidate: dict[str, Any],
    symbol_cfg: AutoTradeSymbolConfig,
) -> dict[str, Any]:
    for key in (
        "net_stop_loss_bps",
        "stop_loss_bps",
        "expected_risk",
        "expected_risk_bps",
    ):
        if _safe_float(candidate.get(key), 0.0) > 0:
            return candidate

    reward_risk = candidate.get("edge_reward_risk") or {}
    if isinstance(reward_risk, dict):
        if _safe_float(reward_risk.get("loss_risk_floor_bps"), 0.0) > 0:
            return candidate

    price = _safe_float(symbol_cfg.price or symbol_cfg.order_price, 0.0)
    stop_loss = _safe_float(symbol_cfg.stop_loss, 0.0)
    if price <= 0 or stop_loss <= 0 or stop_loss >= price:
        return candidate

    return {
        **candidate,
        "stop_loss_bps": (price - stop_loss) / price * 10_000.0,
    }


def _apply_candidate_position_sizing(
    *,
    symbol_cfg: AutoTradeSymbolConfig,
    candidate: dict[str, Any],
) -> AutoTradeSymbolConfig:
    if not _candidate_has_edge_for_sizing(candidate):
        return symbol_cfg

    account_equity = _safe_float(symbol_cfg.account_equity, 0.0)
    risk_per_trade = _safe_float(symbol_cfg.risk_per_trade, 0.0)
    if account_equity <= 0 or risk_per_trade <= 0:
        return symbol_cfg

    sizing_candidate = _candidate_with_symbol_stop_risk(candidate, symbol_cfg)
    regime_gate = candidate.get("regime_gate") or {}
    regime_multiplier = 1.0
    if isinstance(regime_gate, dict):
        regime_multiplier = _safe_float(regime_gate.get("position_multiplier"), 1.0)
    try:
        sizing = _position_value_from_edge_and_risk(
            candidate=sizing_candidate,
            account_equity=account_equity,
            risk_per_trade=risk_per_trade,
            regime_position_multiplier=regime_multiplier,
        )
    except Exception:
        return symbol_cfg

    position_value = _safe_float(sizing.get("position_value"), 0.0)
    if position_value <= 0:
        if _safe_float(sizing.get("regime_position_multiplier"), 1.0) <= 0:
            candidate["position_sizing"] = sizing
            return symbol_cfg.model_copy(
                update={
                    "cash_available": 0.0,
                    "position_size": 0.0,
                }
            )
        return symbol_cfg

    candidate["position_sizing"] = sizing

    existing_cash = symbol_cfg.cash_available
    cash_cap = position_value
    if existing_cash is not None:
        cash_cap = min(cash_cap, _safe_float(existing_cash, position_value))

    return symbol_cfg.model_copy(
        update={
            "cash_available": round(cash_cap, 4),
            "position_size": round(cash_cap / account_equity, 6),
        }
    )


def _entry_symbols_from_scanner_candidates(
    *,
    req: AutoTradeStartRequest,
    active_candidates: list[dict[str, Any]],
    open_symbols: set[str],
) -> list[AutoTradeSymbolConfig]:
    now = _now()
    hurdle_rate = _scanner_hurdle_rate_for_mode(req.execution_mode)
    try:
        regime_gate = regime_gate_for_mode(
            execution_mode=req.execution_mode,
            base_hurdle_bps=hurdle_rate,
        )
    except Exception:
        if req.execution_mode == "live":
            return []
        regime_gate = {
            "enabled": True,
            "approved": True,
            "status": "error",
            "regime": "unknown",
            "base_hurdle_bps": round(hurdle_rate, 4),
            "hurdle_adjustment_bps": 0.0,
            "regime_adjusted_hurdle_bps": round(hurdle_rate, 4),
            "position_multiplier": 1.0,
            "message": "regime gate unavailable; paper neutral pass",
        }

    effective_hurdle_rate = _safe_float(
        regime_gate.get("regime_adjusted_hurdle_bps"),
        hurdle_rate,
    )

    if req.execution_mode != "paper" and regime_gate.get("approved") is False:
        return []

    adjusted_rows: list[dict[str, Any]] = []

    for candidate in active_candidates:
        symbol = str(candidate.get("symbol") or "")
        if not symbol or symbol in open_symbols:
            continue

        if candidate.get("status") not in ("READY", "CLAIMED"):
            continue

        if str(candidate.get("expires_at") or "") <= now:
            continue

        base_net_edge = _safe_float(candidate.get("net_edge"), 0.0)

        try:
            portfolio_penalty = _portfolio_penalty_for_candidate(
                candidate=candidate,
                open_symbols=open_symbols,
                execution_mode=req.execution_mode,
            )
        except Exception:
            if req.execution_mode == "live":
                continue
            portfolio_penalty = _neutral_portfolio_penalty(
                enabled=bool(settings.portfolio_penalty_enabled),
            )

        total_penalty = _safe_float(portfolio_penalty.get("total_penalty_bps"), 0.0)
        adjusted_net_edge = base_net_edge - total_penalty

        portfolio_adjusted_candidate = {
            **candidate,
            "base_net_edge": round(base_net_edge, 4),
            "portfolio_adjusted_net_edge": round(adjusted_net_edge, 4),
            "portfolio_penalty": portfolio_penalty,
        }

        try:
            fill_quality = fill_quality_adjustment_for_candidate(
                portfolio_adjusted_candidate,
                execution_mode=req.execution_mode,
            )
        except Exception:
            if req.execution_mode == "live":
                continue
            fill_quality = {
                "status": "error",
                "fill_probability": _safe_float(
                    settings.fill_quality_default_probability,
                    _safe_float(settings.default_fill_probability, 0.95),
                ),
                "fill_slippage_penalty_bps": 0.0,
                "fill_delay_penalty_bps": 0.0,
                "total_fill_quality_penalty_bps": 0.0,
                "approved": True,
                "message": "Fill-quality adjustment unavailable; neutral pass",
            }

        fill_probability = _safe_float(
            fill_quality.get("fill_probability"),
            _safe_float(settings.default_fill_probability, 0.95),
        )
        fill_penalty = _safe_float(
            fill_quality.get("total_fill_quality_penalty_bps"),
            0.0,
        )
        fill_quality_adjusted_edge = adjusted_net_edge * fill_probability - fill_penalty

        if req.execution_mode != "paper" and fill_quality.get("approved") is False:
            continue

        candidate_with_adjustments = {
            **portfolio_adjusted_candidate,
            "pre_fill_quality_edge": round(adjusted_net_edge, 4),
            "fill_quality_adjusted_edge": round(fill_quality_adjusted_edge, 4),
            "fill_quality": fill_quality,
            "regime_gate": regime_gate,
            "effective_hurdle_rate_bps": round(effective_hurdle_rate, 4),
        }

        signal_decay = _signal_decay_for_candidate(
            candidate_with_adjustments,
            execution_mode=req.execution_mode,
        )

        if req.execution_mode != "paper" and signal_decay.get("approved") is False:
            continue

        final_entry_edge = _safe_float(
            signal_decay.get("signal_decay_adjusted_edge"),
            _final_edge_before_signal_decay(candidate_with_adjustments),
        )

        if final_entry_edge <= effective_hurdle_rate:
            continue

        adjusted_rows.append(
            {
                **candidate_with_adjustments,
                "signal_decay": signal_decay,
                "final_entry_edge": round(final_entry_edge, 4),
            }
        )

    adjusted_rows = sorted(
        adjusted_rows,
        key=lambda item: (
            _safe_float(item.get("final_entry_edge"), 0.0),
            _safe_float(
                item.get("fill_quality_adjusted_edge"),
                _safe_float(
                    item.get("portfolio_adjusted_net_edge"),
                    _safe_float(item.get("net_edge"), 0.0),
                ),
            ),
            _safe_float(item.get("composite_score"), 0.0),
            -_safe_int(item.get("rank"), 999),
        ),
        reverse=True,
    )

    symbols: list[AutoTradeSymbolConfig] = []
    for candidate in adjusted_rows:
        symbol_cfg = scanner_candidate_to_symbol_config(req, candidate)
        symbol_cfg = symbol_cfg.model_copy(
            update={
                "fill_quality": candidate.get("fill_quality"),
                "regime_gate": candidate.get("regime_gate"),
                "signal_decay": candidate.get("signal_decay"),
                "final_entry_edge": candidate.get("final_entry_edge"),
            }
        )
        symbol_cfg = _apply_candidate_position_sizing(
            symbol_cfg=symbol_cfg,
            candidate=candidate,
        )
        symbols.append(symbol_cfg)

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
        req.execution_mode in ("live", "broker_paper")
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
    if req.execution_mode in ("live", "broker_paper"):
        try:
            sync_result = broker_sync.sync_kis_account()
        except Exception as exc:
            return {
                "status": "blocked",
                "mode": req.execution_mode,
                "message": f"Broker account balance check failed; no orders will be attempted: {exc}",
            }
        cash_available = _to_float(sync_result.get("total_cash"))
        account_equity = _to_float(sync_result.get("total_value")) or req.account_equity
        if cash_available is None:
            return {
                "status": "blocked",
                "mode": req.execution_mode,
                "message": "Broker account cash balance is unavailable; no orders will be attempted.",
                "broker_sync": sync_result,
            }
        return {
            "status": "ready",
            "mode": req.execution_mode,
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
        symbol_position_size = _to_float(symbol_cfg.position_size)
        if symbol_position_size is not None:
            max_allocation = min(
                max_allocation,
                account_equity * max(0.0, float(symbol_position_size)),
            )
        symbol_cash_cap = _to_float(symbol_cfg.cash_available)
        if symbol_cash_cap is not None:
            max_allocation = min(max_allocation, max(0.0, float(symbol_cash_cap)))
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
        with connect_sqlite(path, row_factory=True) as conn:
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


def _record_fill_quality_for_result(
    *,
    result: dict[str, Any],
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
) -> dict[str, Any] | None:
    try:
        payload: dict[str, Any] = {
            **result,
            "symbol": symbol_cfg.symbol,
            "side": "sell" if symbol_cfg.requested_action == "exit" else "buy",
            "requested_action": symbol_cfg.requested_action,
            "decision_price": getattr(symbol_cfg, "decision_price", None),
            "order_price": getattr(symbol_cfg, "order_price", None),
            "quantity": getattr(symbol_cfg, "quantity", None),
            "requested_quantity": getattr(symbol_cfg, "quantity", None),
            "execution_mode": req.execution_mode,
            "source": "auto_trading",
        }

        execution = result.get("execution")
        if isinstance(execution, dict):
            paper_result = execution.get("paper_result")
            if isinstance(paper_result, dict):
                payload.update(
                    {
                        "filled_price": (
                            paper_result.get("fill_price")
                            or paper_result.get("effective_price")
                        ),
                        "filled_quantity": paper_result.get("quantity"),
                        "slippage_bps": paper_result.get("slippage_bps"),
                        "paper_result": paper_result,
                    }
                )
            order_state_result = execution.get("order_state")
            if isinstance(order_state_result, dict):
                payload.update(
                    {
                        "filled_quantity": (
                            order_state_result.get("filled_quantity")
                            or payload.get("filled_quantity")
                        ),
                        "order_state": order_state_result,
                    }
                )

        recording = record_fill_quality_event(payload)
        result["fill_quality_recording"] = recording
        return recording
    except Exception:
        return None


def _exit_position_snapshot(
    *,
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
) -> dict[str, Any]:
    if symbol_cfg.requested_action != "exit":
        return {}
    try:
        return _open_positions(req.execution_mode).get(symbol_cfg.symbol) or {}
    except Exception:
        return {}


def _order_result_indicates_realized_exit(
    *,
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
    result: dict[str, Any],
) -> bool:
    if symbol_cfg.requested_action != "exit":
        return False

    execution = result.get("execution")
    if not isinstance(execution, dict):
        return False

    if req.execution_mode == "paper":
        paper_result = execution.get("paper_result")
        if not isinstance(paper_result, dict):
            return False
        return (
            str(paper_result.get("order_status") or "").upper()
            in {"FILLED", "PARTIALLY_FILLED"}
            and _safe_float(paper_result.get("quantity"), 0.0) > 0
        )

    order_state_result = execution.get("order_state")
    if not isinstance(order_state_result, dict):
        return False
    return (
        str(order_state_result.get("state") or "").upper() == "FLAT"
        and _safe_float(order_state_result.get("filled_quantity"), 0.0) > 0
    )


def _bps_from_amount(amount: Any, notional: float | None) -> float | None:
    parsed = _safe_float(amount, 0.0)
    if notional is None or notional <= 0:
        return None
    return abs(parsed) / notional * 10_000.0


def _record_outcome_attribution_for_result(
    *,
    result: dict[str, Any],
    req: AutoTradeStartRequest,
    symbol_cfg: AutoTradeSymbolConfig,
    position_before_exit: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not _order_result_indicates_realized_exit(
        req=req,
        symbol_cfg=symbol_cfg,
        result=result,
    ):
        return None

    try:
        position = position_before_exit or {}
        execution = result.get("execution") if isinstance(result.get("execution"), dict) else {}
        paper_result = (
            execution.get("paper_result")
            if isinstance(execution.get("paper_result"), dict)
            else {}
        )
        order_state_result = (
            execution.get("order_state")
            if isinstance(execution.get("order_state"), dict)
            else {}
        )

        entry_price = _to_float(position.get("avg_price"))
        if entry_price is None and req.execution_mode == "paper":
            entry_price = _to_float(position.get("current_price"))

        exit_price = (
            _to_float(result.get("exit_price"))
            or _to_float(result.get("filled_price"))
            or _to_float(paper_result.get("fill_price"))
            or _to_float(paper_result.get("effective_price"))
            or _to_float(result.get("price"))
        )

        quantity = (
            _to_float(paper_result.get("quantity"))
            or _to_float(order_state_result.get("filled_quantity"))
            or _to_float(result.get("quantity"))
            or _to_float(symbol_cfg.quantity)
        )

        notional = None
        if entry_price is not None and quantity is not None:
            notional = entry_price * quantity

        cost_breakdown = paper_result.get("cost_breakdown")
        if not isinstance(cost_breakdown, dict):
            cost_breakdown = {}

        commission_bps = _bps_from_amount(cost_breakdown.get("commission"), notional)
        tax_bps = _bps_from_amount(cost_breakdown.get("tax"), notional)
        slippage_cost_bps = _bps_from_amount(
            cost_breakdown.get("slippage_cost"),
            notional,
        )
        if slippage_cost_bps is None:
            slippage_cost_bps = abs(_safe_float(paper_result.get("slippage_bps"), 0.0))

        trading_cost_bps = None
        if commission_bps is not None or tax_bps is not None:
            trading_cost_bps = (commission_bps or 0.0) + (tax_bps or 0.0)

        recording = record_outcome_attribution(
            {
                **result,
                "symbol": symbol_cfg.symbol,
                "execution_mode": req.execution_mode,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "entry_time": (
                    result.get("entry_time")
                    or position.get("opened_at")
                    or position.get("updated_at")
                    or position.get("synced_at")
                ),
                "exit_time": result.get("exit_time") or result.get("closed_at") or _now(),
                "trading_cost_bps": trading_cost_bps,
                "slippage_cost_bps": slippage_cost_bps,
                "net_edge": getattr(symbol_cfg, "net_edge", None),
                "portfolio_adjusted_net_edge": result.get("portfolio_adjusted_net_edge"),
                "fill_quality_adjusted_edge": result.get("fill_quality_adjusted_edge"),
                "final_entry_edge": result.get("final_entry_edge")
                or getattr(symbol_cfg, "final_entry_edge", None),
                "portfolio_penalty": result.get("portfolio_penalty"),
                "fill_quality": result.get("fill_quality")
                or getattr(symbol_cfg, "fill_quality", None),
                "signal_decay": result.get("signal_decay")
                or getattr(symbol_cfg, "signal_decay", None),
                "regime_gate": result.get("regime_gate")
                or getattr(symbol_cfg, "regime_gate", None),
                "position_sizing": result.get("position_sizing"),
                "paper_result": paper_result,
                "order_state": order_state_result,
                "source": "auto_trading_close",
            }
        )
        result["outcome_attribution"] = recording
        return recording
    except Exception:
        return None


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
        position_before_exit = _exit_position_snapshot(req=req, symbol_cfg=symbol_cfg)
        symbol_cfg = _apply_order_sizing_defaults(
            req,
            symbol_cfg,
            price,
            price_result.get("price_data") or {},
        )
        orderbook_check: dict[str, Any] | None = None
        if (
            settings.pretrade_orderbook_check_enabled
            and symbol_cfg.requested_action != "exit"
        ):
            order_value = None
            if symbol_cfg.quantity and price:
                order_value = float(symbol_cfg.quantity) * float(price)
            elif symbol_cfg.cash_available:
                order_value = float(symbol_cfg.cash_available)

            orderbook = _fetch_orderbook_neutral(symbol_cfg.symbol)
            orderbook_check = _orderbook_feature_summary(
                orderbook,
                order_value=order_value,
            )

            if (
                req.execution_mode != "paper"
                and orderbook_check.get("approved") is False
            ):
                return {
                    "symbol": symbol_cfg.symbol,
                    "status": "blocked",
                    "message": str(
                        orderbook_check.get("message") or "Orderbook check failed"
                    ),
                    "orderbook_check": orderbook_check,
                }

        preview_req = _to_preview_request(symbol_cfg, price)
        preview = create_order_preview(preview_req)
        result: dict[str, Any] = {
            "symbol": symbol_cfg.symbol,
            "status": preview["status"],
            "price": price,
            "price_source": price_result["source"],
            "preview": preview,
        }
        if symbol_cfg.fill_quality is not None:
            result["fill_quality"] = symbol_cfg.fill_quality
        if symbol_cfg.regime_gate is not None:
            result["regime_gate"] = symbol_cfg.regime_gate
        if symbol_cfg.signal_decay is not None:
            result["signal_decay"] = symbol_cfg.signal_decay
        if symbol_cfg.final_entry_edge is not None:
            result["final_entry_edge"] = symbol_cfg.final_entry_edge
        if orderbook_check is not None:
            result["orderbook_check"] = orderbook_check
        if preview["status"] != "pending":
            result["message"] = preview["message"]
            _record_fill_quality_for_result(
                result=result,
                req=req,
                symbol_cfg=symbol_cfg,
            )
            return result

        if req.execution_mode == "paper":
            if not req.auto_confirm_paper:
                result["message"] = "Paper preview created but auto_confirm_paper is false"
                _record_fill_quality_for_result(
                    result=result,
                    req=req,
                    symbol_cfg=symbol_cfg,
                )
                return result
            result["execution"] = confirm_order_preview(
                OrderConfirmRequest(
                    preview_id=preview["preview_id"],
                    preview_token=preview["preview_token"] or "",
                    execution_mode="paper",
                )
            )
            result["status"] = result["execution"]["status"]
            _record_fill_quality_for_result(
                result=result,
                req=req,
                symbol_cfg=symbol_cfg,
            )
            _record_outcome_attribution_for_result(
                result=result,
                req=req,
                symbol_cfg=symbol_cfg,
                position_before_exit=position_before_exit,
            )
            return result

        if req.execution_mode == "broker_paper":
            result["execution"] = execute_broker_paper_order(
                _to_live_order_request(
                    req=req,
                    symbol_cfg=symbol_cfg,
                    preview=preview,
                    price=price,
                    session_id=session_id,
                )
            )
            result["status"] = result["execution"]["status"]
            _record_fill_quality_for_result(
                result=result,
                req=req,
                symbol_cfg=symbol_cfg,
            )
            _record_outcome_attribution_for_result(
                result=result,
                req=req,
                symbol_cfg=symbol_cfg,
                position_before_exit=position_before_exit,
            )
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
        _record_fill_quality_for_result(
            result=result,
            req=req,
            symbol_cfg=symbol_cfg,
        )
        _record_outcome_attribution_for_result(
            result=result,
            req=req,
            symbol_cfg=symbol_cfg,
            position_before_exit=position_before_exit,
        )
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
    if req.execution_mode == "live" and not req.live_confirm_token:
        raise AutoTradingError("live_confirm_token is required for live auto-trading")
    side = str(preview["side"]).lower()
    return LiveOrderRequest(
        symbol=symbol_cfg.symbol,
        market=symbol_cfg.market,
        broker_provider=req.broker_provider,
        strategy_type=symbol_cfg.strategy_type,
        risk_level=symbol_cfg.risk_level,
        side=side,
        order_type="limit",
        price=float(preview["price"]),
        quantity=int(preview["quantity"]),
        confirm_token=req.live_confirm_token or "broker_paper",
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
    if mode in ("live", "broker_paper"):
        return _open_live_positions()
    return _open_paper_positions()


def _open_paper_positions() -> dict[str, dict[str, Any]]:
    path = settings.storage_path(paper_trading.DEFAULT_DB_PATH)
    if not path.exists():
        return {}
    try:
        with connect_sqlite(path, row_factory=True) as conn:
            paper_trading.initialize_db(conn)
            rows = conn.execute(
                """
                SELECT symbol, name, market, sector, quantity,
                       avg_price, avg_price AS current_price, opened_at, updated_at
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
        with connect_sqlite(path, row_factory=True) as conn:
            conn.executescript(broker_sync.SCHEMA_SQL)
            broker_sync._ensure_column(conn, "broker_positions", "opened_at", "TEXT")
            rows = conn.execute(
                """
                SELECT symbol, name, quantity, avg_price, current_price, opened_at, synced_at
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
    if req.execution_mode == "broker_paper":
        probe_symbol = req.symbols[0].symbol if req.symbols else "005930"
        safety = broker_paper_safety_check(req=req, symbol=probe_symbol)
        if not safety.get("approved"):
            raise AutoTradingError(
                str(
                    safety.get("broker_submit_block_reason")
                    or "broker_paper safety check failed"
                ),
                status_code=403,
            )


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
