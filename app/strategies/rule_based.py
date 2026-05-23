from __future__ import annotations

from typing import Any


def build_strategy_decision(
    pipeline_result: dict[str, Any],
    requested_action: str = "auto",
    risk_level: str | None = None,
) -> dict[str, Any]:
    """
    Convert a pipeline result into a conservative order candidate.

    This layer intentionally does not call a broker. It only decides whether the
    current analysis is strong enough to preview an entry or exit candidate.
    """
    scores = pipeline_result.get("scores") or {}
    final_score = float(scores.get("final_score", 0))
    risk_score = float(scores.get("risk_score", 100))
    confidence = float(pipeline_result.get("confidence", 0))
    final_grade = pipeline_result.get("final_grade")
    risk_level = risk_level or pipeline_result.get("risk_level", "medium")
    thresholds = _thresholds_for(risk_level)
    entry_signal = bool(pipeline_result.get("entry_signal"))
    exit_signal = bool(pipeline_result.get("exit_signal"))

    blocking_reasons: list[str] = []
    action = "hold"
    side = None
    signal_type = None

    if requested_action not in ("auto", "entry", "exit"):
        blocking_reasons.append("requested_action must be auto, entry, or exit")
    elif requested_action == "exit":
        action = "exit"
        side = "SELL"
        signal_type = "exit"
    elif requested_action in ("auto", "exit") and exit_signal:
        action = "exit"
        side = "SELL"
        signal_type = "exit"
    elif requested_action in ("auto", "entry"):
        entry_blockers = _entry_blockers(
            final_grade=final_grade,
            entry_signal=entry_signal,
            final_score=final_score,
            risk_score=risk_score,
            confidence=confidence,
            thresholds=thresholds,
        )
        if not entry_blockers:
            action = "entry"
            side = "BUY"
            signal_type = "entry"
        else:
            blocking_reasons.extend(entry_blockers)

    if action == "hold" and not blocking_reasons:
        blocking_reasons.append("No actionable entry or exit signal")

    return {
        "action": action,
        "side": side,
        "signal_type": signal_type,
        "approved": action in ("entry", "exit"),
        "blocking_reasons": blocking_reasons,
        "final_grade": final_grade,
        "final_score": final_score,
        "risk_score": risk_score,
        "confidence": confidence,
        "risk_level": risk_level,
        "thresholds": thresholds,
        "summary": pipeline_result.get("summary"),
    }


def _entry_blockers(
    final_grade: str | None,
    entry_signal: bool,
    final_score: float,
    risk_score: float,
    confidence: float,
    thresholds: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if final_grade not in thresholds["allowed_grades"]:
        blockers.append(f"Final grade is not in {thresholds['allowed_grades']}")
    if not entry_signal:
        blockers.append("Pipeline entry_signal is false")
    if final_score < thresholds["min_final_score"]:
        blockers.append(f"Final score is below {thresholds['min_final_score']}")
    if risk_score > thresholds["max_risk_score"]:
        blockers.append(f"Risk score is above {thresholds['max_risk_score']}")
    if confidence < thresholds["min_confidence"]:
        blockers.append(f"Confidence is below {thresholds['min_confidence']}")
    return blockers


def _thresholds_for(risk_level: str) -> dict[str, Any]:
    profiles = {
        "low": {
            "allowed_grades": ["공격"],
            "min_final_score": 78,
            "max_risk_score": 45,
            "min_confidence": 0.75,
        },
        "medium": {
            "allowed_grades": ["공격"],
            "min_final_score": 75,
            "max_risk_score": 50,
            "min_confidence": 0.7,
        },
        "high": {
            "allowed_grades": ["공격"],
            "min_final_score": 70,
            "max_risk_score": 60,
            "min_confidence": 0.6,
        },
    }
    return profiles.get(risk_level, profiles["medium"])
