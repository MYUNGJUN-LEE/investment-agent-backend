from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

from app.config import settings
from app.trading import edge_calibration, paper_trading
from app.trading import auto_trading_store


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
    paths = resolved_storage_paths()
    auto_sessions = _auto_sessions(paths["auto_trading_db_path"])
    latest_session = auto_sessions["latest_session"]
    active_sessions = auto_sessions["active_sessions"]
    mode = _resolve_execution_mode(execution_mode, latest_session)
    mode_flags = execution_mode_flags(mode)

    planned_entry_count = _planned_entry_count(latest_session)
    paper_orders_count = _table_count(
        paths["paper_trading_db_path"],
        "paper_orders",
    )
    order_state = _order_state_counts(paths["order_state_db_path"])
    broker_executions_count = _table_count(
        paths["broker_sync_db_path"],
        "broker_order_executions",
    )
    scanner = _scanner_execution_state(paths["universe_scanner_db_path"])
    edge_status = _edge_metric_status(
        calibration_path=paths["edge_calibration_db_path"],
        latest_candidate=scanner.get("latest_execution_candidate"),
        sample_limit=sample_limit,
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

    return {
        "status": "success",
        **mode_flags,
        "DATA_DIR": str(settings.storage_root()) if settings.storage_root() else None,
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
        "planned_entry_count": planned_entry_count,
        "submitted_order_count": submitted_order_count,
        "paper_orders_count": paper_orders_count,
        "broker_executions_count": broker_executions_count,
        "broker_order_id_count": order_state["broker_order_id_count"],
        "pending_order_intent_count": order_state["pending_order_intent_count"],
        "claimed_candidate_count": scanner["claimed_candidate_count"],
        "ready_candidate_count": scanner["ready_candidate_count"],
        "latest_execution_candidate": scanner.get("latest_execution_candidate"),
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
        "win_rates": win_rates,
        **_flat_win_rate_fields(win_rates),
        **edge_status["public_fields"],
    }


def startup_log_payload(
    *,
    execution_mode: str | None = None,
    auto_trading_worker_enabled: bool | None = None,
    scanner_worker_enabled: bool | None = None,
) -> dict[str, Any]:
    mode = str(execution_mode or "paper").lower()
    flags = execution_mode_flags(mode)
    paths = resolved_storage_paths()
    return {
        "DATA_DIR": str(settings.storage_root()) if settings.storage_root() else None,
        "universe_scanner_db_path": str(paths["universe_scanner_db_path"]),
        "edge_calibration_db_path": str(paths["edge_calibration_db_path"]),
        "auto_trading_db_path": str(paths["auto_trading_db_path"]),
        "order_state_db_path": str(paths["order_state_db_path"]),
        "auto_trading_worker_enabled": bool(auto_trading_worker_enabled),
        "scanner_worker_enabled": bool(scanner_worker_enabled),
        "execution_mode": flags["execution_mode"],
        "kis_is_paper": flags["kis_is_paper"],
        "submits_to_broker": flags["submits_to_broker"],
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


def execution_mode_flags(execution_mode: str | None = None) -> dict[str, Any]:
    mode = str(execution_mode or "paper").lower()
    return {
        "execution_mode": mode,
        "broker_provider": "KIS",
        "kis_is_paper": bool(settings.kis_is_paper),
        "submits_to_broker": mode == "live",
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
) -> str:
    if execution_mode:
        return str(execution_mode).lower()
    if latest_session and latest_session.get("execution_mode"):
        return str(latest_session["execution_mode"]).lower()
    return "paper"


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
    return {
        "active_sessions": active_sessions,
        "latest_session": latest[0] if latest else None,
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
    }
    if not path.exists():
        return default
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            if not _table_exists(conn, "order_intents"):
                return default
            submitted_order_count = _scalar_int(
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
    except sqlite3.Error:
        return default
    return {
        "submitted_order_count": submitted_order_count,
        "broker_order_id_count": broker_order_id_count,
        "pending_order_intent_count": pending_order_intent_count,
    }


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
                SELECT scan_id, scan_time, symbol, rank, status, reason, raw_json
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


def _edge_metric_status(
    *,
    calibration_path: Path,
    latest_candidate: dict[str, Any] | None,
    sample_limit: int,
) -> dict[str, Any]:
    public = {
        "gate_calibration_db_path": str(calibration_path),
        "dashboard_edge_calibration_db_path": str(calibration_path),
        "gate_metric_snapshot_timestamp": None,
        "edge_metric_updated_at": None,
        "gate_sample_count": None,
        "dashboard_edge_sample_count": None,
        "candidate_reason_sample_count": None,
        "candidate_reason_required_sample_count": None,
        "stale_reason": False,
    }
    summary: dict[str, Any] = {}
    if calibration_path.exists():
        try:
            gate = edge_calibration.edge_entry_gate(
                calibration_db_path=calibration_path,
            )
            public["gate_sample_count"] = gate.get("sample_count")
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
        "paper_order_win_rate": paper.get("win_rate"),
        "paper_order_win_rate_display": paper.get("display"),
        "paper_order_sample_count": paper.get("sample_count"),
        "broker_execution_win_rate": broker.get("win_rate"),
        "broker_execution_win_rate_display": broker.get("display"),
        "broker_execution_sample_count": broker.get("sample_count"),
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
) -> str:
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
