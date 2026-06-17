from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

from app.config import settings
from app.brokers.kis_client import KIS_PAPER_BASE_URL, KisClient
from app.trading import edge_calibration, paper_trading
from app.trading import auto_trading_store
from app.trading import order_state as order_state_store


logger = logging.getLogger(__name__)


_SAMPLE_COUNT_PATTERN = re.compile(r"sample_count\s+(\d+)\s*/\s*(\d+)", re.I)


def trading_status_snapshot(
    *,
    execution_mode: str | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Return a read-only execution/dashboard status snapshot.

    This intentionally does not initialize missing execution databases. A missing
    DB is itself a diagnostic signal for the dashboard and `/trading/status`.
    """
    storage = settings.storage_status()
    paths = resolved_storage_paths()
    auto_sessions = _auto_sessions(paths["auto_trading_db_path"])
    latest_session = auto_sessions["latest_session"]
    active_sessions = auto_sessions["active_sessions"]
    mode_info = settings.execution_mode_status(execution_mode)
    provider_info = settings.broker_provider_status()
    mode = _resolve_execution_mode(
        execution_mode,
        latest_session,
        configured_mode=mode_info,
    )
    broker_provider = (
        provider_info["broker_provider"]
        if provider_info.get("configured_broker_provider")
        else _resolve_broker_provider(latest_session)
    )
    mode_flags = execution_mode_flags(mode, broker_provider=broker_provider)
    broker_safety = _broker_submit_static_status(
        execution_mode=mode,
        broker_provider=broker_provider,
    )
    kis_token = _kis_token_status(mode=mode, broker_provider=broker_provider)
    kis_account = _kis_account_status(mode=mode, broker_provider=broker_provider)
    if broker_safety["broker_submit_blocked"]:
        mode_flags["submits_to_broker"] = False

    planned_entry_count = _planned_entry_count(latest_session)
    paper_orders_count = _table_count(
        paths["paper_trading_db_path"],
        "paper_orders",
    )
    order_state = _order_state_counts(paths["order_state_db_path"])
    broker_risk = order_state_store.broker_paper_order_risk_snapshot(
        paths["order_state_db_path"]
    )
    broker_executions_count = _table_count(
        paths["broker_sync_db_path"],
        "broker_order_executions",
    )
    scanner = _scanner_execution_state(paths["universe_scanner_db_path"])
    edge_status = _edge_metric_status(
        calibration_path=paths["edge_calibration_db_path"],
        latest_candidate=scanner.get("latest_execution_candidate"),
        sample_limit=sample_limit,
        execution_mode=mode,
    )
    win_rates = _win_rate_status(
        edge_status.get("sample_summary") or {},
        paper_orders_count=paper_orders_count,
        broker_executions_count=broker_executions_count,
        execution_mode=mode,
    )

    submitted_order_count = order_state["submitted_order_count"]
    order_status = _execution_order_status(
        submitted_order_count=submitted_order_count,
        paper_orders_count=paper_orders_count,
        planned_entry_count=planned_entry_count,
        claimed_candidate_count=scanner["claimed_candidate_count"],
        uses_internal_paper_orders=mode_flags["uses_internal_paper_orders"],
        latest_broker_order_event=order_state.get("latest_broker_order_event"),
    )
    paper_zero_reason = _paper_orders_zero_reason(
        uses_internal_paper_orders=mode_flags["uses_internal_paper_orders"],
        paper_orders_count=paper_orders_count,
        active_session_count=len(active_sessions),
        latest_session=latest_session,
        planned_entry_count=planned_entry_count,
        claimed_candidate_count=scanner["claimed_candidate_count"],
        order_status=order_status,
    )
    if paper_zero_reason:
        logger.warning("paper_orders_count is zero: %s", paper_zero_reason)
    session_mode = str((latest_session or {}).get("execution_mode") or "") or None
    session_mismatch = _session_mode_mismatch(
        resolved_mode=mode,
        active_sessions=active_sessions,
    )
    broker_enabled = bool(
        mode_flags["submits_to_broker"]
        and not broker_safety["broker_submit_blocked"]
        and not session_mismatch["session_mode_mismatch"]
    )
    guard_counts = _broker_guard_counts(latest_session)
    post_claim = _post_claim_diagnostics(latest_session)

    return {
        "status": "success",
        **mode_flags,
        **mode_info,
        **provider_info,
        "resolved_execution_mode": mode,
        "active_session_execution_mode": session_mode,
        "session_mode_mismatch": session_mismatch["session_mode_mismatch"],
        "session_mode_mismatch_reason": session_mismatch["session_mode_mismatch_reason"],
        "requires_session_restart": session_mismatch["requires_session_restart"],
        "broker_submit_enabled": broker_enabled,
        "DATA_DIR": storage.get("resolved_data_dir"),
        "configured_data_dir": storage.get("configured_data_dir"),
        "resolved_data_dir": storage.get("resolved_data_dir"),
        "data_dir_writable": storage.get("data_dir_writable"),
        "data_dir_is_persistent": storage.get("data_dir_is_persistent"),
        "data_dir_warning": storage.get("data_dir_warning"),
        "storage_root_fallback_used": storage.get("storage_root_fallback_used"),
        "db_paths": {key: str(value) for key, value in paths.items()},
        "auto_trading_db_path": str(paths["auto_trading_db_path"]),
        "order_state_db_path": str(paths["order_state_db_path"]),
        "universe_scanner_db_path": str(paths["universe_scanner_db_path"]),
        "edge_calibration_db_path": str(paths["edge_calibration_db_path"]),
        "paper_trading_db_path": str(paths["paper_trading_db_path"]),
        "broker_sync_db_path": str(paths["broker_sync_db_path"]),
        "auto_trading_db_missing": not paths["auto_trading_db_path"].exists(),
        "order_state_db_missing": not paths["order_state_db_path"].exists(),
        "active_session_count": len(active_sessions),
        "latest_session": latest_session,
        "latest_session_status": (latest_session or {}).get("status"),
        "account_key": (latest_session or {}).get("account_key"),
        "planned_entry_count": planned_entry_count,
        "submitted_order_count": submitted_order_count,
        "broker_paper_order_count": submitted_order_count,
        "paper_orders_count": paper_orders_count,
        "broker_executions_count": broker_executions_count,
        "broker_order_id_count": order_state["broker_order_id_count"],
        "pending_order_intent_count": order_state["pending_order_intent_count"],
        "latest_broker_order_event": order_state.get("latest_broker_order_event"),
        "last_broker_submit_at": order_state.get("last_broker_submit_at"),
        "last_broker_submit_error": order_state.get("last_broker_submit_error"),
        "last_broker_sync_at": _last_broker_sync_at(paths["broker_sync_db_path"]),
        **kis_token,
        **kis_account,
        "broker_submit_blocked": broker_safety["broker_submit_blocked"],
        "broker_submit_block_reason": broker_safety["broker_submit_block_reason"],
        "broker_submit_block_code": broker_safety.get("broker_submit_block_code"),
        "last_broker_sync_error": _last_broker_sync_error(paths["broker_sync_db_path"]),
        "broker_submit_status": _broker_submit_status(
            broker_safety=broker_safety,
            latest_event=order_state.get("latest_broker_order_event"),
        ),
        "broker_execution_status": (
            "synced" if broker_executions_count > 0 else "none"
        ),
        "candidate_status": _candidate_status(scanner),
        "planner_status": "planned" if planned_entry_count > 0 else "not_planned",
        "claimed_candidate_count": scanner["claimed_candidate_count"],
        "ready_candidate_count": scanner["ready_candidate_count"],
        "latest_execution_candidate": scanner.get("latest_execution_candidate"),
        "latest_post_claim_diagnostics": post_claim["latest_post_claim_diagnostics"],
        "claimed_no_order_count": post_claim["claimed_no_order_count"],
        "claimed_no_order_reasons": post_claim["claimed_no_order_reasons"],
        "claimed_order_diagnostics": post_claim["claimed_order_diagnostics"],
        "order_status": order_status,
        "paper_orders_zero_reason": paper_zero_reason,
        "entry_planner_running": planned_entry_count > 0,
        "order_manager_running": (
            paper_orders_count > 0
            or submitted_order_count > 0
            or broker_executions_count > 0
        ),
        "execution_layer_issue": bool(
            mode_flags["uses_internal_paper_orders"] and paper_orders_count == 0
        ),
        **guard_counts,
        **broker_risk,
        "win_rates": win_rates,
        **_flat_win_rate_fields(win_rates),
        "candidate_label_mae_edge_error_bps": (
            (edge_status.get("sample_summary") or {}).get("summary") or {}
        ).get("mae_net_edge_error_bps"),
        "broker_paper_avg_realized_net_edge_bps": edge_status[
            "public_fields"
        ].get("broker_paper_fill_avg_realized_net_edge_bps"),
        **edge_status["public_fields"],
    }


def startup_log_payload(
    *,
    execution_mode: str | None = None,
    broker_provider: str | None = None,
    auto_trading_worker_enabled: bool | None = None,
    scanner_worker_enabled: bool | None = None,
) -> dict[str, Any]:
    storage = settings.storage_status()
    paths = resolved_storage_paths()
    latest_session = None
    if execution_mode is None and paths["auto_trading_db_path"].exists():
        latest_session = _auto_sessions(paths["auto_trading_db_path"]).get("latest_session")
    mode_info = settings.execution_mode_status(execution_mode)
    provider_info = settings.broker_provider_status(broker_provider)
    mode = str(
        _resolve_execution_mode(
            execution_mode,
            latest_session,
            configured_mode=mode_info,
        )
    ).lower()
    provider = str(provider_info["broker_provider"]).lower()
    flags = execution_mode_flags(mode, broker_provider=provider)
    safety = _broker_submit_static_status(
        execution_mode=mode,
        broker_provider=provider,
    )
    return {
        "DATA_DIR": storage.get("resolved_data_dir"),
        "configured_data_dir": storage.get("configured_data_dir"),
        "resolved_data_dir": storage.get("resolved_data_dir"),
        "data_dir_writable": storage.get("data_dir_writable"),
        "data_dir_is_persistent": storage.get("data_dir_is_persistent"),
        "data_dir_warning": storage.get("data_dir_warning"),
        "storage_root_fallback_used": storage.get("storage_root_fallback_used"),
        **mode_info,
        "resolved_execution_mode": mode,
        **provider_info,
        "universe_scanner_db_path": str(paths["universe_scanner_db_path"]),
        "edge_calibration_db_path": str(paths["edge_calibration_db_path"]),
        "auto_trading_db_path": str(paths["auto_trading_db_path"]),
        "order_state_db_path": str(paths["order_state_db_path"]),
        "auto_trading_worker_enabled": bool(auto_trading_worker_enabled),
        "scanner_worker_enabled": bool(scanner_worker_enabled),
        "execution_mode": flags["execution_mode"],
        "broker_provider": flags["broker_provider"],
        "kis_is_paper": flags["kis_is_paper"],
        "submits_to_broker": False
        if safety["broker_submit_blocked"]
        else flags["submits_to_broker"],
        "uses_internal_paper_orders": flags["uses_internal_paper_orders"],
        "broker_submit_enabled": bool(
            flags["submits_to_broker"] and not safety["broker_submit_blocked"]
        ),
        "broker_submit_safety_check_result": safety,
    }


def print_startup_log(**kwargs: Any) -> None:
    payload = startup_log_payload(**kwargs)
    print(
        json.dumps(
            {"event": "trading_startup_config", **payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def execution_mode_flags(
    execution_mode: str | None = None,
    *,
    broker_provider: str | None = None,
) -> dict[str, Any]:
    mode = str(execution_mode or "paper").lower()
    provider = str(broker_provider or "kis").lower()
    return {
        "execution_mode": mode,
        "broker_provider": provider,
        "kis_is_paper": bool(settings.kis_is_paper),
        "submits_to_broker": mode in ("broker_paper", "live"),
        "uses_internal_paper_orders": mode == "paper",
    }


def resolved_storage_paths() -> dict[str, Path]:
    return {
        "universe_scanner_db_path": settings.storage_path(
            settings.universe_scanner_db_path
        ),
        "edge_calibration_db_path": settings.storage_path(
            settings.edge_calibration_db_path
        ),
        "auto_trading_db_path": settings.storage_path(settings.auto_trading_db_path),
        "order_state_db_path": settings.storage_path(settings.order_state_db_path),
        "paper_trading_db_path": settings.storage_path(paper_trading.DEFAULT_DB_PATH),
        "broker_sync_db_path": settings.storage_path(settings.broker_sync_db_path),
    }


def _resolve_execution_mode(
    execution_mode: str | None,
    latest_session: dict[str, Any] | None,
    configured_mode: dict[str, Any] | None = None,
) -> str:
    if execution_mode:
        return str(execution_mode).lower()
    configured_mode = configured_mode or settings.execution_mode_status()
    if configured_mode.get("configured_execution_mode"):
        return str(configured_mode["resolved_execution_mode"]).lower()
    if latest_session and latest_session.get("execution_mode"):
        return str(latest_session["execution_mode"]).lower()
    return "paper"


def _resolve_broker_provider(latest_session: dict[str, Any] | None) -> str:
    payload = (latest_session or {}).get("request_payload") or {}
    if latest_session:
        return str(payload.get("broker_provider") or latest_session.get("broker_provider") or "kis").lower()
    return "kis"


def _auto_sessions(auto_db_path: Path) -> dict[str, Any]:
    if not auto_db_path.exists():
        return {"active_sessions": [], "latest_session": None}
    try:
        active_sessions = auto_trading_store.list_sessions(
            status="active",
            limit=50,
            db_path=auto_db_path,
        )
        latest = auto_trading_store.list_sessions(limit=1, db_path=auto_db_path)
    except sqlite3.Error as exc:
        return {
            "active_sessions": [],
            "latest_session": {"status": "error", "last_error": str(exc)},
        }
    latest_session = active_sessions[0] if active_sessions else (latest[0] if latest else None)
    return {
        "active_sessions": active_sessions,
        "latest_session": latest_session,
    }


def _planned_entry_count(latest_session: dict[str, Any] | None) -> int:
    if not latest_session:
        return 0
    return _planned_entry_count_from_results(latest_session.get("last_results") or [])


def _planned_entry_count_from_results(results: Any) -> int:
    total = 0
    if not isinstance(results, list):
        return 0
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("planned_entry_count") is not None:
            total += _to_int(item.get("planned_entry_count"))
        elif isinstance(item.get("planned_entries"), list):
            total += len(item["planned_entries"])
        if isinstance(item.get("results"), list):
            total += _planned_entry_count_from_results(item["results"])
    return total


def _order_state_counts(path: Path) -> dict[str, int]:
    default = {
        "submitted_order_count": 0,
        "broker_order_id_count": 0,
        "pending_order_intent_count": 0,
        "latest_broker_order_event": None,
        "last_broker_submit_at": None,
        "last_broker_submit_error": None,
    }
    if not path.exists():
        return default
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            intent_submitted_count = 0
            broker_order_id_count = 0
            pending_order_intent_count = 0
            event_submitted_count = 0
            latest_event = None
            last_submit_at = None
            last_submit_error = None
            if _table_exists(conn, "order_intents"):
                intent_submitted_count = _scalar_int(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM order_intents
                    WHERE status IN ('SUBMITTED', 'PARTIALLY_FILLED', 'FILLED')
                    """,
                )
                broker_order_id_count = _scalar_int(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM order_intents
                    WHERE COALESCE(broker_order_no, '') <> ''
                    """,
                )
                pending_order_intent_count = _scalar_int(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM order_intents
                    WHERE status IN ('PENDING', 'SUBMITTED', 'PARTIALLY_FILLED')
                    """,
                )
            if not _table_exists(conn, "broker_order_events"):
                return {
                    "submitted_order_count": intent_submitted_count,
                    "broker_order_id_count": broker_order_id_count,
                    "pending_order_intent_count": pending_order_intent_count,
                    "latest_broker_order_event": None,
                    "last_broker_submit_at": None,
                    "last_broker_submit_error": None,
                }
            event_submitted_count = _scalar_int(
                conn,
                """
                SELECT COUNT(*)
                FROM broker_order_events
                WHERE order_status IN (
                    'submitted', 'accepted', 'unknown_pending',
                    'filled', 'partially_filled', 'rejected'
                )
                """,
            )
            broker_order_id_count = max(
                broker_order_id_count,
                _scalar_int(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM broker_order_events
                    WHERE COALESCE(broker_order_id, '') <> ''
                    """,
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM broker_order_events
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                latest_event = _broker_event_row_to_dict(row)
                last_submit_at = latest_event.get("created_at")
            error_row = conn.execute(
                """
                SELECT reject_reason, broker_response_message
                FROM broker_order_events
                WHERE order_status = 'rejected'
                   OR COALESCE(reject_reason, '') <> ''
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            if error_row:
                last_submit_error = error_row[0] or error_row[1]
            return {
                "submitted_order_count": max(
                    intent_submitted_count,
                    event_submitted_count,
                ),
                "broker_order_id_count": broker_order_id_count,
                "pending_order_intent_count": pending_order_intent_count,
                "latest_broker_order_event": latest_event,
                "last_broker_submit_at": last_submit_at,
                "last_broker_submit_error": last_submit_error,
            }
    except sqlite3.Error:
        return default


def _broker_event_row_to_dict(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        data = dict(row)
    else:
        return {}
    data["kis_is_paper"] = bool(data.get("kis_is_paper"))
    data["raw_response"] = _parse_json(data.pop("raw_response_json", None), {})
    return data


def _last_broker_sync_at(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            if _table_exists(conn, "broker_order_executions"):
                row = conn.execute(
                    "SELECT MAX(synced_at) FROM broker_order_executions"
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
            if _table_exists(conn, "broker_balance_snapshots"):
                row = conn.execute(
                    "SELECT MAX(created_at) FROM broker_balance_snapshots"
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
    except sqlite3.Error:
        return None
    return None


def _last_broker_sync_error(path: Path) -> str | None:
    return None


def _session_mode_mismatch(
    *,
    resolved_mode: str,
    active_sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatched = [
        session
        for session in active_sessions
        if str(session.get("execution_mode") or "paper").lower() != resolved_mode
    ]
    if not mismatched:
        return {
            "session_mode_mismatch": False,
            "session_mode_mismatch_reason": None,
            "requires_session_restart": False,
        }
    return {
        "session_mode_mismatch": True,
        "session_mode_mismatch_reason": "active_session_execution_mode_differs_from_config",
        "requires_session_restart": True,
        "mismatched_session_ids": [session.get("session_id") for session in mismatched],
    }


def _broker_guard_counts(latest_session: dict[str, Any] | None) -> dict[str, Any]:
    counts = {
        "dedupe_blocked_count": 0,
        "open_order_blocked_count": 0,
        "already_position_blocked_count": 0,
        "daily_order_limit_blocked_count": 0,
    }
    if not latest_session:
        return counts
    stack = list(latest_session.get("last_results") or [])
    while stack:
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("results"), list):
            stack.extend(item["results"])
        code = str(
            item.get("broker_submit_block_code")
            or item.get("broker_order_guard_code")
            or ""
        )
        message = str(item.get("message") or "")
        if code in {"duplicate_scan_symbol_side", "symbol_cooldown_active"}:
            counts["dedupe_blocked_count"] += 1
        if code in {"open_broker_order_exists", "open_order_intent_exists"}:
            counts["open_order_blocked_count"] += 1
        if code == "already_position_exists":
            counts["already_position_blocked_count"] += 1
        if code in {
            "daily_order_limit_exceeded",
            "daily_symbol_order_limit_exceeded",
            "daily_notional_limit_exceeded",
        }:
            counts["daily_order_limit_blocked_count"] += 1
        if "duplicate" in message.lower():
            counts["dedupe_blocked_count"] += 1
    return counts


def _kis_token_status(*, mode: str, broker_provider: str) -> dict[str, Any]:
    defaults = {
        "kis_token_cached": False,
        "kis_token_status": "not_applicable",
        "kis_token_expires_at": None,
        "kis_token_seconds_to_expiry": None,
        "kis_token_last_refresh_at": None,
        "kis_token_last_refresh_attempt_at": None,
        "kis_token_last_refresh_error": None,
        "kis_token_refresh_blocked_by_rate_limit": False,
        "kis_token_next_refresh_allowed_at": None,
    }
    if str(broker_provider or "").lower() != "kis":
        return defaults
    if str(mode or "").lower() not in {"broker_paper", "live"}:
        return defaults
    try:
        return {
            **defaults,
            **KisClient(is_paper=bool(settings.kis_is_paper)).token_status(),
        }
    except Exception as exc:
        return {
            **defaults,
            "kis_token_status": "error",
            "kis_token_last_refresh_error": str(exc),
        }


def _kis_account_status(*, mode: str, broker_provider: str) -> dict[str, Any]:
    defaults = {
        "kis_account_rate_limited": False,
        "kis_account_next_probe_allowed_at": None,
        "kis_account_last_probe_at": None,
        "kis_account_last_probe_operation": None,
        "kis_account_last_probe_error": None,
        "kis_account_last_success_at": None,
        "kis_account_cache_enabled": False,
        "kis_account_cached_operation_count": 0,
    }
    if str(broker_provider or "").lower() != "kis":
        return defaults
    if str(mode or "").lower() not in {"broker_paper", "live"}:
        return defaults
    try:
        return {
            **defaults,
            **KisClient(is_paper=bool(settings.kis_is_paper)).account_status(),
        }
    except Exception as exc:
        return {
            **defaults,
            "kis_account_last_probe_error": str(exc),
        }


def _broker_submit_static_status(
    *,
    execution_mode: str,
    broker_provider: str,
) -> dict[str, Any]:
    if execution_mode != "broker_paper":
        return {
            "broker_submit_blocked": False,
            "broker_submit_block_reason": None,
            "broker_submit_block_code": None,
        }
    provider = str(broker_provider or "").lower()
    if provider != "kis":
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": f"broker_provider must be kis, got {provider}",
            "broker_submit_block_code": "broker_provider_not_kis",
        }
    if not settings.kis_is_paper:
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": "broker_paper requires KIS_IS_PAPER=true",
            "broker_submit_block_code": "kis_is_paper_false",
        }
    token_status = _kis_token_status(
        mode=execution_mode,
        broker_provider=broker_provider,
    )
    if (
        token_status.get("kis_token_refresh_blocked_by_rate_limit")
        and not token_status.get("kis_token_cached")
    ):
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": "kis_token_unavailable_rate_limited",
            "broker_submit_block_code": "kis_token_unavailable_rate_limited",
            "kis_token": token_status,
        }
    account_status = _kis_account_status(
        mode=execution_mode,
        broker_provider=broker_provider,
    )
    if account_status.get("kis_account_rate_limited"):
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": "kis_account_rate_limited",
            "broker_submit_block_code": "kis_account_rate_limited",
            "kis_account": account_status,
        }
    storage = settings.storage_status()
    if not storage.get("data_dir_writable") or not storage.get("data_dir_is_persistent"):
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": "persistent_order_storage_unavailable",
            "broker_submit_block_code": "persistent_order_storage_unavailable",
            "storage": {
                "resolved_data_dir": storage.get("resolved_data_dir"),
                "data_dir_writable": storage.get("data_dir_writable"),
                "data_dir_is_persistent": storage.get("data_dir_is_persistent"),
                "data_dir_warning": storage.get("data_dir_warning"),
                "storage_root_fallback_used": storage.get("storage_root_fallback_used"),
            },
        }
    missing = []
    if not settings.kis_app_key:
        missing.append("KIS_APP_KEY")
    if not settings.kis_app_secret:
        missing.append("KIS_APP_SECRET")
    if not settings.kis_account_no:
        missing.append("KIS_ACCOUNT_NO")
    if not settings.kis_account_product_code:
        missing.append("KIS_ACCOUNT_PRODUCT_CODE")
    if missing:
        return {
            "broker_submit_blocked": True,
            "broker_submit_block_reason": "Missing KIS mock account config: "
            + ", ".join(missing),
            "broker_submit_block_code": "kis_config_missing",
        }
    return {
        "broker_submit_blocked": False,
        "broker_submit_block_reason": None,
        "broker_submit_block_code": None,
        "kis_paper_endpoint": KIS_PAPER_BASE_URL,
    }


def _broker_submit_status(
    *,
    broker_safety: dict[str, Any],
    latest_event: dict[str, Any] | None,
) -> str:
    if broker_safety.get("broker_submit_blocked"):
        return "blocked"
    if latest_event:
        return str(latest_event.get("order_status") or "unknown_pending")
    return "idle"


def _candidate_status(scanner: dict[str, Any]) -> str:
    if scanner.get("claimed_candidate_count"):
        return "claimed"
    if scanner.get("ready_candidate_count"):
        return "ready"
    return "none"


def _scanner_execution_state(path: Path) -> dict[str, Any]:
    result = {
        "claimed_candidate_count": 0,
        "ready_candidate_count": 0,
        "latest_execution_candidate": None,
    }
    if not path.exists():
        return result
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            if not _table_exists(conn, "scanner_candidates"):
                return result
            result["claimed_candidate_count"] = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM scanner_candidates WHERE status = 'CLAIMED'",
            )
            result["ready_candidate_count"] = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM scanner_candidates WHERE status = 'READY'",
            )
            row = conn.execute(
                """
                SELECT scan_id, scan_time, symbol, rank, status, reason,
                       claimed_by_worker, claimed_at, fresh_quote_used,
                       fresh_quote_age_seconds, cached_snapshot_age_seconds,
                       exclusion_reason, raw_json
                FROM scanner_candidates
                WHERE status IN ('READY', 'CLAIMED')
                ORDER BY scan_time DESC, rank ASC
                LIMIT 1
                """
            ).fetchone()
            if row:
                item = dict(row)
                item["raw"] = _parse_json(item.pop("raw_json", None), {})
                result["latest_execution_candidate"] = item
    except sqlite3.Error:
        return result
    return result


def _post_claim_diagnostics(latest_session: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    if latest_session:
        stack = list(latest_session.get("last_results") or [])
        while stack:
            item = stack.pop()
            if not isinstance(item, dict):
                continue
            trace = item.get("post_claim_diagnostics")
            if isinstance(trace, dict):
                diagnostics.append(trace)
            nested_scan = item.get("claimed_order_diagnostics")
            if isinstance(nested_scan, list):
                diagnostics.extend(
                    row for row in nested_scan if isinstance(row, dict)
                )
            if isinstance(item.get("results"), list):
                stack.extend(item["results"])
    diagnostics.sort(key=lambda row: str(row.get("claim_time") or ""), reverse=True)
    reason_counts: dict[str, int] = {}
    for row in diagnostics:
        reason = row.get("claimed_no_order_reason")
        if reason:
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "latest_post_claim_diagnostics": diagnostics[0] if diagnostics else None,
        "claimed_order_diagnostics": diagnostics[:20],
        "claimed_no_order_count": sum(reason_counts.values()),
        "claimed_no_order_reasons": reason_counts,
    }


def _edge_metric_status(
    *,
    calibration_path: Path,
    latest_candidate: dict[str, Any] | None,
    sample_limit: int,
    execution_mode: str | None,
) -> dict[str, Any]:
    public = {
        "gate_calibration_db_path": str(calibration_path),
        "dashboard_edge_calibration_db_path": str(calibration_path),
        "gate_metric_snapshot_timestamp": None,
        "edge_metric_updated_at": None,
        "gate_metric_source": None,
        "gate_sample_count": None,
        "all_sample_count": None,
        "top10_sample_count": None,
        "top10_avg_return_bps": None,
        "top10_realized_net_edge_bps": None,
        "top10_win_rate": None,
        "top10_expectancy_bps": None,
        "top10_predicted_edge_bps": None,
        "gate_avg_return_bps": None,
        "gate_win_rate": None,
        "gate_realized_net_edge_bps": None,
        "gate_expectancy_bps": None,
        "gate_predicted_edge_bps": None,
        "ignored_all_sample_gate_metrics": False,
        "dashboard_edge_sample_count": None,
        "candidate_reason_sample_count": None,
        "candidate_reason_required_sample_count": None,
        "stale_reason": False,
        "broker_paper_bootstrap_enabled": bool(
            settings.broker_paper_bootstrap_enabled
        ),
        "broker_paper_calibration_source": settings.broker_paper_calibration_source,
        "broker_paper_candidate_label_gate_mode": (
            settings.broker_paper_candidate_label_gate_mode
        ),
        "candidate_label_gate_failed": None,
        "candidate_label_gate_hard_blocking": None,
        "broker_paper_fill_sample_count": 0,
        "broker_paper_oos_fill_sample_count": 0,
        "broker_paper_fill_win_rate": None,
        "broker_paper_fill_profit_factor": None,
        "broker_paper_fill_avg_realized_net_edge_bps": None,
        "broker_paper_fill_mae_edge_error_bps": None,
        "broker_paper_min_fill_samples": settings.broker_paper_min_fill_samples,
        "broker_paper_min_oos_fill_samples": settings.broker_paper_min_oos_fill_samples,
        "broker_paper_fill_gate_ready": False,
        "broker_paper_fill_gate_hard_blocking": False,
        "broker_paper_fill_gate_blocked": False,
        "calibration_gate_mode": None,
        "net_edge_aggregate_splits": {},
        "all_observed_net_edge_bps": None,
        "executable_only_net_edge_bps": None,
        "risk_rejected_net_edge_bps": None,
        "top_rank_executable_net_edge_bps": None,
        "false_positive_net_edge_impact_bps": None,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "severe_false_positive_count": 0,
    }
    summary: dict[str, Any] = {}
    try:
        gate = edge_calibration.edge_entry_gate(
            calibration_db_path=calibration_path,
            execution_mode=execution_mode,
        )
    except TypeError:
        try:
            gate = edge_calibration.edge_entry_gate(
                calibration_db_path=calibration_path,
            )
        except Exception as exc:
            gate = None
            public["gate_error"] = str(exc)
    except Exception as exc:
        gate = None
        public["gate_error"] = str(exc)
    if gate:
        public["gate_sample_count"] = gate.get("sample_count")
        for key in (
            "broker_paper_bootstrap_enabled",
            "broker_paper_calibration_source",
            "broker_paper_candidate_label_gate_mode",
            "candidate_label_gate_failed",
            "candidate_label_gate_hard_blocking",
            "broker_paper_fill_sample_count",
            "broker_paper_oos_fill_sample_count",
            "broker_paper_fill_win_rate",
            "broker_paper_fill_profit_factor",
            "broker_paper_fill_avg_realized_net_edge_bps",
            "broker_paper_fill_mae_edge_error_bps",
            "broker_paper_min_fill_samples",
            "broker_paper_min_oos_fill_samples",
            "broker_paper_fill_gate_ready",
            "broker_paper_fill_gate_hard_blocking",
            "broker_paper_fill_gate_blocked",
            "calibration_gate_mode",
            "gate_metric_source",
            "all_sample_count",
            "top10_sample_count",
            "top10_avg_return_bps",
            "top10_realized_net_edge_bps",
            "top10_win_rate",
            "top10_expectancy_bps",
            "top10_predicted_edge_bps",
            "gate_avg_return_bps",
            "gate_win_rate",
            "gate_realized_net_edge_bps",
            "gate_expectancy_bps",
            "gate_predicted_edge_bps",
            "ignored_all_sample_gate_metrics",
        ):
            if key in gate:
                public[key] = gate.get(key)
    if calibration_path.exists():
        try:
            public["gate_metric_snapshot_timestamp"] = _latest_edge_run_created_at(
                calibration_path
            )
        except Exception as exc:
            public["gate_error"] = str(exc)
        try:
            summary = edge_calibration.get_edge_training_sample_summary(
                calibration_db_path=calibration_path,
                limit=sample_limit,
            )
            public["dashboard_edge_sample_count"] = (
                (summary.get("summary") or {}).get("sample_count")
                or summary.get("sample_count")
            )
            splits = summary.get("net_edge_aggregate_splits") or {}
            executable_split = splits.get("executable_candidates_only") or {}
            public["net_edge_aggregate_splits"] = splits
            public["all_observed_net_edge_bps"] = (
                (splits.get("all_observed_candidates") or {}).get(
                    "total_realized_net_edge_bps"
                )
            )
            public["executable_only_net_edge_bps"] = executable_split.get(
                "total_realized_net_edge_bps"
            )
            public["risk_rejected_net_edge_bps"] = (
                (splits.get("risk_rejected_candidates") or {}).get(
                    "total_realized_net_edge_bps"
                )
            )
            public["top_rank_executable_net_edge_bps"] = (
                (splits.get("top_rank_executable_only") or {}).get(
                    "total_realized_net_edge_bps"
                )
            )
            public["false_positive_net_edge_impact_bps"] = executable_split.get(
                "false_positive_net_edge_impact_bps"
            )
            public["false_positive_count"] = (
                executable_split.get("false_positive_count") or 0
            )
            public["false_positive_rate"] = (
                executable_split.get("false_positive_rate") or 0.0
            )
            public["severe_false_positive_count"] = (
                executable_split.get("severe_false_positive_count") or 0
            )
        except Exception as exc:
            public["dashboard_edge_error"] = str(exc)
        public["edge_metric_updated_at"] = _edge_metric_updated_at(calibration_path)

    reason_counts = _candidate_reason_sample_count(latest_candidate)
    public.update(reason_counts)
    public["stale_reason"] = _is_stale_candidate_reason(
        candidate=latest_candidate,
        reason_sample_count=reason_counts["candidate_reason_sample_count"],
        gate_sample_count=public.get("gate_sample_count"),
        dashboard_sample_count=public.get("dashboard_edge_sample_count"),
        edge_metric_updated_at=public.get("edge_metric_updated_at"),
    )
    return {
        "public_fields": public,
        "sample_summary": summary,
    }


def _latest_edge_run_created_at(path: Path) -> str | None:
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            if not _table_exists(conn, "edge_calibration_runs"):
                return None
            row = conn.execute(
                """
                SELECT created_at
                FROM edge_calibration_runs
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] is not None else None


def _edge_metric_updated_at(path: Path) -> str | None:
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            if not _table_exists(conn, "edge_calibration_meta"):
                return _latest_edge_run_created_at(path)
            for key in (
                "last_success_at",
                "last_top10_performance_at",
                "last_attempt_at",
            ):
                row = conn.execute(
                    """
                    SELECT value
                    FROM edge_calibration_meta
                    WHERE key = ?
                    """,
                    (key,),
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
    except sqlite3.Error:
        return None
    return _latest_edge_run_created_at(path)


def _candidate_reason_sample_count(
    candidate: dict[str, Any] | None,
) -> dict[str, int | None]:
    if not candidate:
        return {
            "candidate_reason_sample_count": None,
            "candidate_reason_required_sample_count": None,
        }
    reason = str(candidate.get("reason") or "")
    match = _SAMPLE_COUNT_PATTERN.search(reason)
    if not match:
        return {
            "candidate_reason_sample_count": None,
            "candidate_reason_required_sample_count": None,
        }
    return {
        "candidate_reason_sample_count": int(match.group(1)),
        "candidate_reason_required_sample_count": int(match.group(2)),
    }


def _is_stale_candidate_reason(
    *,
    candidate: dict[str, Any] | None,
    reason_sample_count: int | None,
    gate_sample_count: Any,
    dashboard_sample_count: Any,
    edge_metric_updated_at: str | None,
) -> bool:
    if reason_sample_count is not None:
        comparable_counts = [
            _to_nullable_int(gate_sample_count),
            _to_nullable_int(dashboard_sample_count),
        ]
        if any(
            count is not None and count != reason_sample_count
            for count in comparable_counts
        ):
            return True
    if candidate and edge_metric_updated_at:
        scan_time = _parse_iso(candidate.get("scan_time"))
        metric_time = _parse_iso(edge_metric_updated_at)
        if scan_time and metric_time and scan_time < metric_time:
            return True
    return False


def _win_rate_status(
    sample_summary: dict[str, Any],
    *,
    paper_orders_count: int,
    broker_executions_count: int,
    execution_mode: str,
) -> dict[str, Any]:
    units = sample_summary.get("unit_performance") or {}
    candidate = _unit_win_rate(units.get(edge_calibration.CANDIDATE_LABEL_UNIT))
    paper = _unit_win_rate(units.get(edge_calibration.PAPER_ORDER_UNIT))
    broker = _unit_win_rate(units.get(edge_calibration.ACTUAL_BROKER_FILL_UNIT))

    if paper_orders_count <= 0:
        paper = {**paper, "win_rate": None, "display": "N/A", "status": "empty"}
    if broker_executions_count <= 0:
        broker = {**broker, "win_rate": None, "display": "N/A", "status": "empty"}

    actual = paper if execution_mode == "paper" else broker
    return {
        "candidate_label": candidate,
        "paper_order": paper,
        "broker_execution": broker,
        "actual_trading": actual,
        "candidate_label_win_rate_is_actual_trading_win_rate": False,
    }


def _unit_win_rate(unit: dict[str, Any] | None) -> dict[str, Any]:
    unit = unit or {}
    win_rate = unit.get("win_rate")
    sample_count = _to_int(unit.get("sample_count"))
    return {
        "sample_count": sample_count,
        "win_rate": win_rate,
        "display": _format_percent(win_rate),
        "status": unit.get("status") or ("ready" if sample_count else "empty"),
        "unit": unit.get("unit"),
        "avg_return_bps": unit.get("avg_return_bps"),
        "avg_realized_net_edge_bps": unit.get("avg_realized_net_edge_bps"),
        "mae_net_edge_error_bps": unit.get("mae_net_edge_error_bps"),
        "profit_factor": unit.get("profit_factor"),
    }


def _flat_win_rate_fields(win_rates: dict[str, Any]) -> dict[str, Any]:
    candidate = win_rates["candidate_label"]
    paper = win_rates["paper_order"]
    broker = win_rates["broker_execution"]
    actual = win_rates["actual_trading"]
    return {
        "candidate_label_win_rate": candidate.get("win_rate"),
        "candidate_label_win_rate_display": candidate.get("display"),
        "candidate_label_sample_count": candidate.get("sample_count"),
        "candidate_label_avg_return_bps": candidate.get("avg_return_bps"),
        "candidate_label_mae_edge_error_bps": candidate.get("mae_net_edge_error_bps"),
        "internal_paper_order_sample_count": paper.get("sample_count"),
        "internal_paper_order_win_rate": paper.get("win_rate"),
        "paper_order_win_rate": paper.get("win_rate"),
        "paper_order_win_rate_display": paper.get("display"),
        "paper_order_sample_count": paper.get("sample_count"),
        "paper_order_avg_realized_net_edge_bps": paper.get(
            "avg_realized_net_edge_bps"
        ),
        "broker_execution_win_rate": broker.get("win_rate"),
        "broker_execution_win_rate_display": broker.get("display"),
        "broker_execution_sample_count": broker.get("sample_count"),
        "broker_execution_avg_realized_net_edge_bps": broker.get(
            "avg_realized_net_edge_bps"
        ),
        "actual_trading_win_rate": actual.get("win_rate"),
        "actual_trading_win_rate_display": actual.get("display"),
        "candidate_label_win_rate_is_actual_trading_win_rate": False,
    }


def _execution_order_status(
    *,
    submitted_order_count: int,
    paper_orders_count: int,
    planned_entry_count: int,
    claimed_candidate_count: int,
    uses_internal_paper_orders: bool,
    latest_broker_order_event: dict[str, Any] | None = None,
) -> str:
    if latest_broker_order_event:
        return str(latest_broker_order_event.get("order_status") or "unknown_pending")
    if submitted_order_count > 0:
        return "submitted"
    if uses_internal_paper_orders and paper_orders_count > 0:
        return "paper_order_created"
    if claimed_candidate_count > 0:
        return "not_submitted"
    if planned_entry_count > 0:
        return "planned_not_submitted"
    return "idle"


def _paper_orders_zero_reason(
    *,
    uses_internal_paper_orders: bool,
    paper_orders_count: int,
    active_session_count: int,
    latest_session: dict[str, Any] | None,
    planned_entry_count: int,
    claimed_candidate_count: int,
    order_status: str,
) -> str | None:
    if not uses_internal_paper_orders or paper_orders_count > 0:
        return None
    if active_session_count <= 0:
        return "auto_trading_worker_or_session_not_running"
    if latest_session and latest_session.get("last_error"):
        return f"latest_session_error:{latest_session['last_error']}"
    if planned_entry_count <= 0 and claimed_candidate_count > 0:
        return "candidate_claimed_but_entry_planner_created_no_entries"
    if planned_entry_count <= 0:
        return "entry_planner_created_no_entries"
    if order_status == "planned_not_submitted":
        return "entry_planner_created_entries_but_order_manager_did_not_submit"
    return "paper_order_manager_not_reached"


def _table_count(path: Path, table: str, where: str = "") -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            if not _table_exists(conn, table):
                return 0
            suffix = f" WHERE {where}" if where else ""
            return _scalar_int(conn, f"SELECT COUNT(*) FROM {table}{suffix}")
    except sqlite3.Error:
        return 0


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return bool(row)


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row else 0


def _parse_json(value: str | None, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _to_nullable_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"
