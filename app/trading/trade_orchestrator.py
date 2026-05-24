from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.config import settings
from app.trading import auto_trading, auto_trading_store
from app.trading.edge_calibration import calibrate_edge_model_if_due
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
    initialize_universe_db()
    calibration_result = calibrate_edge_model_if_due()

    sessions = auto_trading_store.list_sessions(status="active", limit=session_limit)
    if not sessions:
        return {
            "status": "idle",
            "worker_id": worker_id,
            "message": "No active auto-trading sessions",
            "edge_calibration": calibration_result,
            "session_count": 0,
            "results": [],
        }

    active_candidates = get_active_scanner_candidates(
        limit=settings.universe_scanner_final_limit,
        include_expired=True,
    )
    if not active_candidates:
        return {
            "status": "idle",
            "worker_id": worker_id,
            "message": "scanner_candidates is empty; no orchestrated exit or entry attempted",
            "edge_calibration": calibration_result,
            "session_count": len(sessions),
            "results": [],
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
            auto_trading_store.record_session_event(
                session_id,
                event_type="orchestrator_completed",
                status=result["status"],
                message=result["message"],
                results=[result],
                update_last_results=True,
            )
            results.append(
                {
                    "session_id": session_id,
                    **result,
                }
            )
        except Exception as exc:
            message = f"Trade orchestrator failed: {exc}"
            auto_trading_store.record_session_event(
                session_id,
                event_type="orchestrator_failed",
                status="error",
                message=message,
                results=[],
                update_last_results=False,
            )
            results.append(
                {
                    "session_id": session_id,
                    "status": "error",
                    "message": message,
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
        "edge_calibration": calibration_result,
        "session_count": len(sessions),
        "candidate_count": len(active_candidates),
        "candidate_symbols": [item.get("symbol") for item in active_candidates],
        "results": results,
    }
