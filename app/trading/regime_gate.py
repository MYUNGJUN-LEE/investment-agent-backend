from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.storage.market_data import get_latest_market_context


_CACHE: dict[str, Any] = {
    "loaded_at": None,
    "context": None,
}


def _now() -> datetime:
    return datetime.now(UTC)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _setting_float(value: Any, default: float) -> float:
    parsed = _to_float(value)
    return default if parsed is None else parsed


def _setting_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _normalize_regime(value: Any) -> str:
    raw = str(value or "").strip().lower()

    if raw in {"bull", "risk_on", "uptrend", "strong", "positive"}:
        return "bull"

    if raw in {"bear", "risk_off", "downtrend", "weak", "negative"}:
        return "bear"

    if raw in {"stress", "crash", "panic"}:
        return "stress"

    return "neutral"


def _cached_market_context() -> dict[str, Any]:
    cache_seconds = max(0, _setting_int(settings.regime_gate_cache_seconds, 60))
    loaded_at = _CACHE.get("loaded_at")

    if (
        loaded_at is not None
        and isinstance(loaded_at, datetime)
        and (_now() - loaded_at).total_seconds() <= cache_seconds
        and isinstance(_CACHE.get("context"), dict)
    ):
        return dict(_CACHE["context"])

    try:
        context = get_latest_market_context() or {}
    except Exception:
        context = {}

    if not isinstance(context, dict):
        context = {}

    _CACHE["loaded_at"] = _now()
    _CACHE["context"] = dict(context)
    return context


def classify_market_regime(context: dict[str, Any]) -> dict[str, Any]:
    if not context:
        return {
            "status": "missing",
            "regime": "unknown",
            "risk_on_score": None,
            "index_return_1d": None,
            "breadth": None,
            "reason": "market context unavailable",
        }

    explicit_regime = _normalize_regime(
        context.get("market_regime")
        or context.get("regime")
        or context.get("trend")
    )

    kospi = context.get("kospi") if isinstance(context.get("kospi"), dict) else {}
    kosdaq = context.get("kosdaq") if isinstance(context.get("kosdaq"), dict) else {}
    raw = context.get("raw") if isinstance(context.get("raw"), dict) else {}

    risk_on_score = _to_float(context.get("risk_on_score"))
    index_return_1d = _first_float(
        context.get("index_return_1d"),
        context.get("kospi_return_1d"),
        context.get("benchmark_return_1d"),
        context.get("kospi_change_pct"),
        context.get("kosdaq_change_pct"),
        kospi.get("change_pct"),
        kosdaq.get("change_pct"),
        raw.get("index_return_1d"),
        raw.get("kospi_return_1d"),
        raw.get("benchmark_return_1d"),
    )
    breadth = _to_float(context.get("breadth"))

    stress_score = _setting_float(settings.regime_gate_stress_risk_on_score, 25.0)
    bull_score = _setting_float(settings.regime_gate_min_risk_on_score_bull, 65.0)
    neutral_score = _setting_float(settings.regime_gate_min_risk_on_score_neutral, 45.0)
    stress_return = _setting_float(
        settings.regime_gate_negative_index_return_1d_stress_pct,
        -2.0,
    )

    if explicit_regime == "stress":
        return {
            "status": "ready",
            "regime": "stress",
            "risk_on_score": risk_on_score,
            "index_return_1d": index_return_1d,
            "breadth": breadth,
            "reason": "explicit stress regime",
        }

    if risk_on_score is not None and risk_on_score <= stress_score:
        return {
            "status": "ready",
            "regime": "stress",
            "risk_on_score": risk_on_score,
            "index_return_1d": index_return_1d,
            "breadth": breadth,
            "reason": f"risk_on_score {risk_on_score:.2f} <= stress threshold {stress_score:.2f}",
        }

    if index_return_1d is not None and index_return_1d <= stress_return:
        return {
            "status": "ready",
            "regime": "stress",
            "risk_on_score": risk_on_score,
            "index_return_1d": index_return_1d,
            "breadth": breadth,
            "reason": f"index_return_1d {index_return_1d:.2f}% <= stress threshold {stress_return:.2f}%",
        }

    if explicit_regime == "bull":
        return {
            "status": "ready",
            "regime": "bull",
            "risk_on_score": risk_on_score,
            "index_return_1d": index_return_1d,
            "breadth": breadth,
            "reason": "explicit bull/risk-on regime",
        }

    if explicit_regime == "bear":
        return {
            "status": "ready",
            "regime": "bear",
            "risk_on_score": risk_on_score,
            "index_return_1d": index_return_1d,
            "breadth": breadth,
            "reason": "explicit bear/risk-off regime",
        }

    if risk_on_score is not None:
        if risk_on_score >= bull_score:
            return {
                "status": "ready",
                "regime": "bull",
                "risk_on_score": risk_on_score,
                "index_return_1d": index_return_1d,
                "breadth": breadth,
                "reason": f"risk_on_score {risk_on_score:.2f} >= bull threshold {bull_score:.2f}",
            }

        if risk_on_score < neutral_score:
            return {
                "status": "ready",
                "regime": "bear",
                "risk_on_score": risk_on_score,
                "index_return_1d": index_return_1d,
                "breadth": breadth,
                "reason": f"risk_on_score {risk_on_score:.2f} < neutral threshold {neutral_score:.2f}",
            }

    return {
        "status": "ready",
        "regime": "neutral",
        "risk_on_score": risk_on_score,
        "index_return_1d": index_return_1d,
        "breadth": breadth,
        "reason": "neutral market regime",
    }


def regime_gate_for_mode(
    *,
    execution_mode: str = "paper",
    base_hurdle_bps: float = 0.0,
) -> dict[str, Any]:
    base_hurdle = _setting_float(base_hurdle_bps, 0.0)

    if not bool(settings.regime_gate_enabled):
        return {
            "enabled": False,
            "approved": True,
            "regime": "disabled",
            "base_hurdle_bps": round(base_hurdle, 4),
            "hurdle_adjustment_bps": 0.0,
            "regime_adjusted_hurdle_bps": round(base_hurdle, 4),
            "position_multiplier": 1.0,
            "message": "regime gate disabled",
        }

    context = _cached_market_context()
    classification = classify_market_regime(context)

    execution_mode = str(execution_mode or "paper").lower()
    paper = execution_mode == "paper"
    regime = str(classification.get("regime") or "unknown").lower()

    approved = True
    message = str(classification.get("reason") or "")

    if classification.get("status") == "missing":
        if paper:
            adjustment = 0.0
            multiplier = 1.0
            message = "market context missing; paper neutral pass"
        else:
            adjustment = _setting_float(
                settings.regime_gate_missing_context_live_hurdle_add_bps,
                15.0,
            )
            multiplier = _setting_float(
                settings.regime_gate_neutral_position_multiplier,
                0.85,
            )
            message = "market context missing; live cautious pass"

        return {
            "enabled": True,
            "approved": True,
            "status": "missing",
            "regime": "unknown",
            "base_hurdle_bps": round(base_hurdle, 4),
            "hurdle_adjustment_bps": round(adjustment, 4),
            "regime_adjusted_hurdle_bps": round(base_hurdle + adjustment, 4),
            "position_multiplier": round(max(0.0, min(1.25, multiplier)), 4),
            "classification": classification,
            "message": message,
        }

    if regime == "bull":
        adjustment = _setting_float(settings.regime_gate_bull_hurdle_add_bps, -10.0)
        multiplier = _setting_float(settings.regime_gate_bull_position_multiplier, 1.0)
    elif regime == "bear":
        adjustment = _setting_float(settings.regime_gate_bear_hurdle_add_bps, 30.0)
        multiplier = _setting_float(settings.regime_gate_bear_position_multiplier, 0.5)
        if execution_mode != "paper" and bool(settings.regime_gate_live_block_bear):
            approved = False
            message = "bear regime blocked live entries"
    elif regime == "stress":
        adjustment = _setting_float(settings.regime_gate_stress_hurdle_add_bps, 80.0)
        multiplier = _setting_float(settings.regime_gate_stress_position_multiplier, 0.0)
        if execution_mode != "paper" and bool(settings.regime_gate_live_block_stress):
            approved = False
            message = "stress regime blocked live entries"
    else:
        adjustment = _setting_float(settings.regime_gate_neutral_hurdle_add_bps, 0.0)
        multiplier = _setting_float(settings.regime_gate_neutral_position_multiplier, 0.85)

    adjusted_hurdle = base_hurdle + adjustment

    return {
        "enabled": True,
        "approved": approved,
        "status": "ready",
        "regime": regime,
        "base_hurdle_bps": round(base_hurdle, 4),
        "hurdle_adjustment_bps": round(adjustment, 4),
        "regime_adjusted_hurdle_bps": round(adjusted_hurdle, 4),
        "position_multiplier": round(max(0.0, min(1.25, multiplier)), 4),
        "classification": classification,
        "message": message,
    }
