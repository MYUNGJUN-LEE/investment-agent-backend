from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.config import settings
from app.trading import auto_trading, auto_trading_store
from app.trading.edge_calibration import (
    calibrate_edge_model_if_due,
    record_fill_adjustment_from_fills,
    refresh_edge_training_samples,
    refresh_top10_performance_if_due,
)
from app.trading.universe_scanner import (
    get_active_scanner_candidates,
    initialize_universe_db,
)


def run_trade_orchestrator_once(
    *,
    worker_id: str | None = None,
    session_limit: int = 20,
) -> dict[str, Any]:
    """Compare current scanner top-10 candidates with real holdings and trade deltas."""
    worker_id = worker_id or f"trade-orchestrator-{uuid4().hex[:8]}"
    auto_trading_store.initialize_auto_trading_db()
    recovered_sessions = auto_trading_store.recover_overdue_active_sessions()
    initialize_universe_db()
    label_refresh = _safe_step("edge_label_refresh", refresh_edge_training_samples)
    calibration_result = _safe_step("edge_calibration", calibrate_edge_model_if_due)
    top10_performance_refresh = _safe_step(
        "top10_performance",
        refresh_top10_performance_if_due,
    )
    fill_adjustment = _safe_step("fill_adjustment", record_fill_adjustment_from_fills)

    sessions = auto_trading_store.list_sessions(status="active", limit=session_limit)
    if not sessions:
        return {
            "status": "idle",
            "worker_id": worker_id,
            "message": "No active auto-trading sessions",
            "label_refresh": label_refresh,
            "top10_performance_refresh": top10_performance_refresh,
            "recovered_session_count": len(recovered_sessions),
            "edge_calibration": calibration_result,
            "fill_adjustment": fill_adjustment,
            "session_count": 0,
            "results": [],
        }

    active_candidates = get_active_scanner_candidates(
        limit=settings.universe_scanner_final_limit,
        include_expired=True,
    )
    if not active_candidates:
        results: list[dict[str, Any]] = []
        for session in sessions:
            session_id = session["session_id"]
            result = {
                "status": "blocked",
                "message": "scanner_candidates is empty; no orchestrated exit or entry attempted",
                "active_candidate_symbols": [],
                "planned_exit_count": 0,
                "planned_entry_count": 0,
                "planned_exits": [],
                "planned_entries": [],
                "entry_gate": None,
                "results": [],
            }
            _complete_orchestrator_session(
                session_id=session_id,
                result=result,
                event_type="orchestrator_completed",
            )
            results.append({"session_id": session_id, **result})
        return {
            "status": "blocked",
            "worker_id": worker_id,
            "message": "scanner_candidates is empty; no orchestrated exit or entry attempted",
            "label_refresh": label_refresh,
            "top10_performance_refresh": top10_performance_refresh,
            "recovered_session_count": len(recovered_sessions),
            "edge_calibration": calibration_result,
            "fill_adjustment": fill_adjustment,
            "session_count": len(sessions),
            "results": results,
        }

    results: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session["session_id"]
        try:
            req = auto_trading_store.load_request(session)
            result = auto_trading.run_orchestrated_candidates_once(
                req,
                active_candidates=active_candidates,
                session_id=session_id,
                execute_entries=settings.trade_orchestrator_execute_entries,
            )
            _complete_orchestrator_session(
                session_id=session_id,
                result=result,
                event_type="orchestrator_completed",
            )
            results.append(
                {
                    "session_id": session_id,
                    **result,
                }
            )
        except Exception as exc:
            message = f"Trade orchestrator failed: {exc}"
            result = {
                "status": "error",
                "message": message,
                "active_candidate_symbols": [],
                "planned_exit_count": 0,
                "planned_entry_count": 0,
                "planned_exits": [],
                "planned_entries": [],
                "entry_gate": None,
                "results": [],
            }
            _complete_orchestrator_session(
                session_id=session_id,
                result=result,
                event_type="orchestrator_failed",
            )
            results.append(
                {
                    "session_id": session_id,
                    **result,
                }
            )

    status = "idle"
    if any(item.get("status") == "executed" for item in results):
        status = "executed"
    elif any(item.get("status") in ("blocked", "error") for item in results):
        status = "blocked"

    return {
        "status": status,
        "worker_id": worker_id,
        "label_refresh": label_refresh,
        "top10_performance_refresh": top10_performance_refresh,
        "recovered_session_count": len(recovered_sessions),
        "edge_calibration": calibration_result,
        "fill_adjustment": fill_adjustment,
        "session_count": len(sessions),
        "candidate_count": len(active_candidates),
        "candidate_symbols": [item.get("symbol") for item in active_candidates],
        "results": results,
    }


def _complete_orchestrator_session(
    *,
    session_id: str,
    result: dict[str, Any],
    event_type: str,
) -> None:
    auto_trading_store.record_session_event(
        session_id,
        event_type=event_type,
        status=str(result.get("status") or "unknown"),
        message=str(result.get("message") or ""),
        results=[result],
        update_last_results=False,
    )


def _safe_step(name: str, fn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {
            "status": "error",
            "step": name,
            "message": str(exc),
        }
