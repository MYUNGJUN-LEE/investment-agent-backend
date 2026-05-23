from __future__ import annotations

from typing import Any

from app.models import PipelineRequest
from app.scoring import (
    score_research_events,
    score_financial_data,
    score_chart_flow,
    calculate_final_score,
    clamp_score,
)
from app.data_sources.opendart import fetch_opendart_disclosures
from app.data_sources.news import fetch_news
from app.data_sources.financials import fetch_financial_data
from app.data_sources.kis import fetch_price_data
from app.data_sources.market_context import fetch_market_context


RISK_PROFILES = {
    "low": {
        "entry_final_score": 78,
        "entry_risk_max": 45,
        "neutral_score": 68,
        "watch_score": 55,
        "change_min": 0,
        "change_max": 5,
        "volume_ratio_min": 1.8,
        "allow_sideways_entry": False,
        "require_above_ma5": True,
        "require_above_ma20": True,
        "intraday_score_min": 65,
        "execution_strength_min": 110,
        "orderbook_imbalance_min": 0.0,
        "spread_pct_max": 0.4,
    },
    "medium": {
        "entry_final_score": 75,
        "entry_risk_max": 50,
        "neutral_score": 65,
        "watch_score": 50,
        "change_min": 0,
        "change_max": 6,
        "volume_ratio_min": 1.5,
        "allow_sideways_entry": False,
        "require_above_ma5": True,
        "require_above_ma20": True,
        "intraday_score_min": 60,
        "execution_strength_min": 100,
        "orderbook_imbalance_min": -0.05,
        "spread_pct_max": 0.6,
    },
    "high": {
        "entry_final_score": 70,
        "entry_risk_max": 60,
        "neutral_score": 60,
        "watch_score": 45,
        "change_min": -0.5,
        "change_max": 8,
        "volume_ratio_min": 1.2,
        "allow_sideways_entry": True,
        "require_above_ma5": False,
        "require_above_ma20": True,
        "intraday_score_min": 55,
        "execution_strength_min": 90,
        "orderbook_imbalance_min": -0.15,
        "spread_pct_max": 0.8,
    },
}


def run_research_agent(req: PipelineRequest) -> dict[str, Any]:
    disclosures = fetch_opendart_disclosures(req.symbol, req.lookback_hours)
    news = fetch_news(req.name or req.symbol, req.lookback_hours)

    events = disclosures + news
    score = score_research_events(events)

    positive_events = [e for e in events if e.get("impact_direction") == "positive"]
    negative_events = [e for e in events if e.get("impact_direction") == "negative"]

    if score >= 65:
        sentiment = "positive"
    elif score <= 40:
        sentiment = "negative"
    elif positive_events and negative_events:
        sentiment = "mixed"
    else:
        sentiment = "neutral"

    data_needed = []
    if not disclosures:
        data_needed.append("OpenDART API key/corp_code 설정 또는 최근 공시 존재 여부 확인")
    if not news:
        data_needed.append("뉴스 API 연결")

    return {
        "research_score": score,
        "overall_sentiment": sentiment,
        "key_events": events[:10],
        "positive_points": [
            e.get("title", "") for e in positive_events[:5]
        ],
        "negative_points": [
            e.get("title", "") for e in negative_events[:5]
        ],
        "uncertain_points": [
            e.get("title", "") for e in events if e.get("impact_direction") == "uncertain"
        ][:5],
        "things_to_verify": data_needed,
        "summary": f"총 {len(events)}개 이벤트를 확인했습니다. 리서치 점수는 {score}점입니다.",
    }


def run_financial_agent(req: PipelineRequest, research_result: dict[str, Any]) -> dict[str, Any]:
    financial_data = fetch_financial_data(req.symbol)
    score = score_financial_data(financial_data)
    metrics = financial_data.get("metrics") or {}
    growth = financial_data.get("growth_metrics") or {}

    data_needed = []
    if financial_data.get("status") in ("missing_opendart_api_key", "missing_corp_code"):
        data_needed.append("OpenDART 재무제표 조회를 위한 API 키와 corp_code 매핑")
    if financial_data.get("error"):
        data_needed.append("OpenDART 재무 API 오류 확인")

    strengths = []
    weaknesses = []
    if _gte(metrics.get("operating_margin"), 10):
        strengths.append("영업이익률 10% 이상")
    if _gte(metrics.get("roe"), 10):
        strengths.append("ROE 10% 이상")
    if _lte(metrics.get("debt_ratio"), 100):
        strengths.append("부채비율 100% 이하")
    if _gte(growth.get("revenue_growth"), 10):
        strengths.append("매출 성장률 10% 이상")

    if _lt(metrics.get("operating_margin"), 0):
        weaknesses.append("영업손실")
    if _lt(metrics.get("net_margin"), 0):
        weaknesses.append("순손실")
    if _gte(metrics.get("debt_ratio"), 250):
        weaknesses.append("부채비율 250% 이상")
    if _lt(growth.get("revenue_growth"), -10):
        weaknesses.append("매출 역성장")

    return {
        "financial_score": score,
        "growth_score": _score_growth(growth),
        "profitability_score": _score_profitability(metrics),
        "stability_score": _score_stability(metrics),
        "valuation_score": None,
        "summary": f"OpenDART 재무 핵심 지표 기반 점수는 {score}점입니다.",
        "strengths": strengths,
        "weaknesses": weaknesses,
        "valuation_view": "unknown",
        "financial_risks": [
            "실시간 컨센서스, PER/PBR, 동종업계 비교는 아직 연결되지 않았습니다.",
            *weaknesses,
        ],
        "data_needed": data_needed,
        "metrics": metrics,
        "growth_metrics": growth,
        "raw_financial_data_preview": _preview(financial_data),
    }


def run_chart_flow_agent(
    req: PipelineRequest,
    research_result: dict[str, Any],
    financial_result: dict[str, Any],
) -> dict[str, Any]:
    price_data = fetch_price_data(req.symbol)
    market_context = fetch_market_context(symbol=req.symbol, persist=True)
    score = score_chart_flow(price_data)

    data_needed = []
    if price_data.get("status") != "connected":
        data_needed.append("KIS 또는 키움 시세 API 연결")
        data_needed.append("현재가, 분봉, 거래량, 거래대금, VWAP 데이터")
    elif price_data.get("optional_errors"):
        data_needed.extend(price_data.get("optional_errors", []))

    change_rate = price_data.get("change_rate")
    volume_ratio = price_data.get("volume_ratio")
    price_position = price_data.get("price_position") or {}
    intraday = price_data.get("intraday") or {}
    profile = _risk_profile(req.risk_level)
    trend = price_data.get("trend")
    trend_ok = trend == "uptrend" or (
        profile["allow_sideways_entry"] and trend == "sideways"
    )
    ma_ok = (
        (not profile["require_above_ma5"] or bool(price_position.get("above_ma5")))
        and (not profile["require_above_ma20"] or bool(price_position.get("above_ma20")))
    )
    intraday_ok = _intraday_ok(intraday, profile)
    market_context_ok = _market_context_entry_ok(market_context, req.risk_level)
    entry_signal = (
        price_data.get("status") == "connected"
        and trend_ok
        and isinstance(change_rate, (int, float))
        and profile["change_min"] < change_rate < profile["change_max"]
        and isinstance(volume_ratio, (int, float))
        and volume_ratio >= profile["volume_ratio_min"]
        and ma_ok
        and intraday_ok
        and market_context_ok
        and not price_data.get("overheated")
    )
    exit_signal = (
        price_data.get("status") == "connected"
        and (
            (isinstance(change_rate, (int, float)) and change_rate <= -3)
            or price_data.get("trend") == "downtrend"
            or _intraday_exit_signal(intraday)
            or _market_context_exit_signal(market_context)
        )
    )
    entry_conditions = _entry_conditions(intraday, price_data, profile)
    avoid_conditions = _avoid_conditions(intraday, price_data)
    _apply_market_context_conditions(
        market_context=market_context,
        market_context_ok=market_context_ok,
        entry_conditions=entry_conditions,
        avoid_conditions=avoid_conditions,
    )
    flow_view = _flow_view(intraday)
    market_view = _market_context_view(market_context)
    if market_view:
        flow_view = f"{flow_view}; {market_view}" if flow_view else market_view

    return {
        "chart_flow_score": score,
        "trend": trend or "unknown",
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "overheated": bool(price_data.get("overheated")),
        "support_levels": price_data.get("support_levels", []),
        "resistance_levels": price_data.get("resistance_levels", []),
        "volume_view": _volume_view(volume_ratio),
        "flow_view": flow_view,
        "entry_conditions": entry_conditions,
        "avoid_conditions": avoid_conditions,
        "stop_loss_candidates": [
            "VWAP 이탈",
            "직전 저점 이탈",
            "-2% 손실 도달",
        ],
        "take_profit_candidates": [
            "+3~5% 구간 분할익절",
            "거래량 둔화",
            "대장주 반락",
        ],
        "data_needed": data_needed,
        "price_data": price_data,
        "market_context": market_context,
    }


def run_devils_advocate_agent(
    req: PipelineRequest,
    research_result: dict[str, Any],
    financial_result: dict[str, Any],
    chart_flow_result: dict[str, Any],
) -> dict[str, Any]:
    risk_score = 50.0

    counterarguments = [
        {
            "argument": "뉴스 또는 공시가 이미 주가에 선반영되었을 가능성",
            "why_it_matters": "단타에서는 좋은 뉴스라도 발표 직후 이미 가격에 반영되면 기대수익보다 손실위험이 커집니다.",
            "risk_level": "medium",
            "what_to_check": "뉴스 발생 전후 주가와 거래대금 변화",
        },
        {
            "argument": "시세 데이터 미연결 상태에서는 진입 타이밍을 확정하기 어렵습니다.",
            "why_it_matters": "거래량, VWAP, 전고점 돌파 확인 없이 진입하면 추격매수 위험이 커집니다.",
            "risk_level": "high",
            "what_to_check": "현재가, 분봉, 거래량, 거래대금, 호가",
        },
        {
            "argument": "재무·밸류에이션 데이터가 부족하면 테마성 급등인지 실적 기반 상승인지 구분하기 어렵습니다.",
            "why_it_matters": "실적 근거가 약한 상승은 급락 전환 가능성이 큽니다.",
            "risk_level": "medium",
            "what_to_check": "최근 실적, 컨센서스, PER/PBR, 동종업계 비교",
        },
        {
            "argument": "시장 지수 하락 시 개별 호재가 묻힐 수 있습니다.",
            "why_it_matters": "단타에서는 종목 재료보다 시장 전체 위험회피가 더 강하게 작용할 수 있습니다.",
            "risk_level": "medium",
            "what_to_check": "KOSPI/KOSDAQ, 선물, 환율",
        },
        {
            "argument": "유동성 부족 종목은 체결과 손절이 불리할 수 있습니다.",
            "why_it_matters": "스프레드가 넓고 거래대금이 적으면 손절 기준이 무너질 수 있습니다.",
            "risk_level": "medium",
            "what_to_check": "거래대금, 호가 스프레드, 체결강도",
        },
    ]

    if chart_flow_result.get("data_needed"):
        risk_score += 12
    if financial_result.get("data_needed"):
        risk_score += 5

    price_data = chart_flow_result.get("price_data") or {}
    intraday_score = (price_data.get("intraday") or {}).get("intraday_score")
    if isinstance(intraday_score, (int, float)) and intraday_score >= 70:
        risk_score -= 8
    if chart_flow_result.get("chart_flow_score", 0) >= 75:
        risk_score -= 5

    research_score = research_result.get("research_score", 50)
    if research_score < 45:
        risk_score += 10

    main_weakness = (
        "KIS 장중 분봉·호가·체결·수급 데이터가 붙었지만, 단타 신호는 종목별 체결 품질과 시장 지수 급변에 계속 민감합니다."
        if not chart_flow_result.get("data_needed")
        else "실시간 시세/수급 데이터가 완전하지 않으면 진입 타이밍 판단은 제한적입니다."
    )

    return {
        "risk_score": clamp_score(risk_score),
        "main_weakness": main_weakness,
        "counterarguments": counterarguments,
        "do_not_enter_conditions": [
            "전고점 돌파 실패",
            "거래대금 감소",
            "시장 지수 급락",
            "호재 뉴스 이후 주가 미반응",
            "손절 기준이 명확하지 않음",
        ],
        "signals_that_thesis_is_wrong": [
            "VWAP 이탈",
            "직전 저점 이탈",
            "대장주 반락",
            "호재 후 거래량 감소",
            "악재성 공시 발생",
        ],
        "risk_mitigation": [
            "신호 발생 후 바로 실매수하지 말고 모의매매 로그부터 쌓기",
            "1회 손실 한도 설정",
            "연속 손절 시 자동 중단",
        ],
    }


def run_final_check_agent(
    req: PipelineRequest,
    research_result: dict[str, Any],
    financial_result: dict[str, Any],
    chart_flow_result: dict[str, Any],
    devils_result: dict[str, Any],
) -> dict[str, Any]:
    research_score = float(research_result.get("research_score", 50))
    financial_score = float(financial_result.get("financial_score", 50))
    chart_flow_score = float(chart_flow_result.get("chart_flow_score", 50))
    risk_score = float(devils_result.get("risk_score", 50))

    final_score = calculate_final_score(
        research_score=research_score,
        financial_score=financial_score,
        chart_flow_score=chart_flow_score,
        risk_score=risk_score,
    )
    profile = _risk_profile(req.risk_level)

    if final_score >= profile["entry_final_score"] and risk_score <= profile["entry_risk_max"]:
        final_grade = "공격"
        entry_signal = True
    elif final_score >= profile["neutral_score"]:
        final_grade = "중립"
        entry_signal = False
    elif final_score >= profile["watch_score"]:
        final_grade = "관망"
        entry_signal = False
    else:
        final_grade = "회피"
        entry_signal = False

    if chart_flow_result.get("price_data", {}).get("status") != "connected":
        entry_signal = False
        final_grade = "관망" if final_grade in ("공격", "중립") else final_grade

    if not chart_flow_result.get("entry_signal", False):
        entry_signal = False

    return {
        "final_grade": final_grade,
        "entry_signal": entry_signal,
        "exit_signal": bool(chart_flow_result.get("exit_signal", False)),
        "confidence": round(final_score / 100, 2),
        "summary": f"최종 점수 {final_score}점. 리스크 레벨 {req.risk_level} 기준 등급은 {final_grade}입니다. 실시간 시세/수급 연결 전에는 자동 진입 신호를 제한합니다.",
        "scores": {
            "research_score": research_score,
            "financial_score": financial_score,
            "chart_flow_score": chart_flow_score,
            "risk_score": risk_score,
            "final_score": final_score,
        },
        "entry_conditions": chart_flow_result.get("entry_conditions", []),
        "avoid_conditions": list(dict.fromkeys(
            chart_flow_result.get("avoid_conditions", [])
            + devils_result.get("do_not_enter_conditions", [])
        )),
        "stop_loss_candidates": chart_flow_result.get("stop_loss_candidates", []),
        "take_profit_candidates": chart_flow_result.get("take_profit_candidates", []),
        "time_exit_rule": "진입 후 30~60분 내 상승 흐름이 없으면 청산 검토",
        "what_to_verify_now": list(dict.fromkeys(
            research_result.get("things_to_verify", [])
            + financial_result.get("data_needed", [])
            + chart_flow_result.get("data_needed", [])
        )),
        "why_not_to_trade": [
            c["argument"] for c in devils_result.get("counterarguments", [])[:5]
        ],
        "final_comment": "이 결과는 자동매매 명령이 아니라 공개 데이터 기반의 매매 후보 체크리스트입니다.",
    }


def _intraday_ok(intraday: dict[str, Any], profile: dict[str, Any]) -> bool:
    intraday_score = intraday.get("intraday_score")
    if not isinstance(intraday_score, (int, float)):
        return False
    if intraday_score < profile["intraday_score_min"]:
        return False

    execution_strength = intraday.get("execution_strength")
    if (
        isinstance(execution_strength, (int, float))
        and execution_strength < profile["execution_strength_min"]
    ):
        return False

    orderbook_imbalance = intraday.get("orderbook_imbalance")
    if (
        isinstance(orderbook_imbalance, (int, float))
        and orderbook_imbalance < profile["orderbook_imbalance_min"]
    ):
        return False

    spread_pct = intraday.get("spread_pct")
    if isinstance(spread_pct, (int, float)) and spread_pct > profile["spread_pct_max"]:
        return False

    return True


def _intraday_exit_signal(intraday: dict[str, Any]) -> bool:
    minute_momentum = intraday.get("minute_momentum_pct")
    execution_strength = intraday.get("execution_strength")
    orderbook_imbalance = intraday.get("orderbook_imbalance")
    spread_pct = intraday.get("spread_pct")

    return bool(
        (
            intraday.get("above_minute_vwap") is False
            and isinstance(minute_momentum, (int, float))
            and minute_momentum <= -1
        )
        or (
            isinstance(execution_strength, (int, float))
            and execution_strength < 70
        )
        or (
            isinstance(orderbook_imbalance, (int, float))
            and orderbook_imbalance <= -0.35
        )
        or (
            isinstance(spread_pct, (int, float))
            and spread_pct > 1.2
        )
    )


def _market_context_entry_ok(
    market_context: dict[str, Any],
    risk_level: str,
) -> bool:
    if market_context.get("status") not in {"connected", "partial"}:
        return True

    risk_on_score = market_context.get("risk_on_score")
    if isinstance(risk_on_score, (int, float)):
        min_score = {"low": 40, "medium": 35, "high": 30}.get(risk_level, 35)
        if risk_on_score < min_score:
            return False

    if market_context.get("market_regime") == "bear" and risk_level != "high":
        return False
    return True


def _market_context_exit_signal(market_context: dict[str, Any]) -> bool:
    if market_context.get("status") not in {"connected", "partial"}:
        return False
    risk_on_score = market_context.get("risk_on_score")
    return bool(
        market_context.get("market_regime") == "bear"
        and isinstance(risk_on_score, (int, float))
        and risk_on_score < 25
    )


def _apply_market_context_conditions(
    market_context: dict[str, Any],
    market_context_ok: bool,
    entry_conditions: list[str],
    avoid_conditions: list[str],
) -> None:
    if market_context.get("status") not in {"connected", "partial"}:
        return

    regime = market_context.get("market_regime", "unknown")
    risk_on_score = market_context.get("risk_on_score")
    context_text = f"Market regime {regime}, risk-on score {risk_on_score}"
    if market_context_ok:
        entry_conditions.append(context_text)
    else:
        avoid_conditions.append(context_text)


def _market_context_view(market_context: dict[str, Any]) -> str:
    if market_context.get("status") not in {"connected", "partial"}:
        return ""
    regime = market_context.get("market_regime", "unknown")
    risk_on_score = market_context.get("risk_on_score")
    return f"market {regime} ({risk_on_score})"


def _flow_view(intraday: dict[str, Any]) -> str:
    parts = []
    execution_strength = intraday.get("execution_strength")
    orderbook_imbalance = intraday.get("orderbook_imbalance")
    smart_money = intraday.get("smart_money_net_buy_5d")

    if isinstance(execution_strength, (int, float)):
        if execution_strength >= 120:
            parts.append(f"체결강도 우위({execution_strength})")
        elif execution_strength < 90:
            parts.append(f"체결강도 약세({execution_strength})")

    if isinstance(orderbook_imbalance, (int, float)):
        if orderbook_imbalance >= 0.1:
            parts.append("매수호가 잔량 우위")
        elif orderbook_imbalance <= -0.1:
            parts.append("매도호가 잔량 우위")

    if isinstance(smart_money, (int, float)):
        if smart_money > 0:
            parts.append("외국인/기관 5일 순매수")
        elif smart_money < 0:
            parts.append("외국인/기관 5일 순매도")

    if not parts:
        return "장중 수급/체결 신호 중립 또는 데이터 부족"
    return ", ".join(parts)


def _entry_conditions(
    intraday: dict[str, Any],
    price_data: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    conditions = [
        f"최종 장중 점수 {profile['intraday_score_min']}점 이상",
        f"체결강도 {profile['execution_strength_min']} 이상 또는 대체 장중 신호 우위",
        "일봉 추세와 이동평균 조건 충족",
    ]

    if intraday.get("above_minute_vwap") is True:
        conditions.append("분봉 VWAP 위 유지")
    if _gte(intraday.get("minute_volume_ratio"), 1.5):
        conditions.append("최근 분봉 거래량 증가")
    if _gte(intraday.get("orderbook_imbalance"), 0.05):
        conditions.append("호가 잔량 매수 우위")
    smart_money = intraday.get("smart_money_net_buy_5d")
    if isinstance(smart_money, (int, float)) and smart_money > 0:
        conditions.append("외국인/기관 누적 순매수 확인")
    if _gte(price_data.get("volume_ratio"), profile["volume_ratio_min"]):
        conditions.append("20일 평균 대비 거래량 조건 충족")

    return list(dict.fromkeys(conditions))


def _avoid_conditions(
    intraday: dict[str, Any],
    price_data: dict[str, Any],
) -> list[str]:
    conditions = [
        "거래량 없는 상승",
        "전고점 돌파 실패",
        "지수 급락",
        "급등 후 장대음봉",
    ]

    if intraday.get("above_minute_vwap") is False:
        conditions.append("분봉 VWAP 이탈")
    if _lt(intraday.get("execution_strength"), 80):
        conditions.append("체결강도 80 미만")
    if _lt(intraday.get("orderbook_imbalance"), -0.2):
        conditions.append("매도호가 잔량 우위")
    if _gte(intraday.get("spread_pct"), 0.8):
        conditions.append("호가 스프레드 과대")
    if _lt(intraday.get("smart_money_net_buy_5d"), 0):
        conditions.append("외국인/기관 누적 순매도")
    if price_data.get("overheated"):
        conditions.append("단기 과열 구간")

    return list(dict.fromkeys(conditions))


def _preview(data: Any, limit: int = 5) -> Any:
    if isinstance(data, dict) and isinstance(data.get("list"), list):
        preview = dict(data)
        preview["list"] = data["list"][:limit]
        preview["list_truncated"] = len(data["list"]) > limit
        return preview
    return data


def _gte(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value >= threshold


def _lte(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value <= threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value < threshold


def _score_growth(growth: dict[str, Any]) -> float | None:
    values = [
        growth.get("revenue_growth"),
        growth.get("operating_income_growth"),
        growth.get("net_income_growth"),
    ]
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return clamp_score(50 + sum(numeric) / len(numeric))


def _score_profitability(metrics: dict[str, Any]) -> float | None:
    values = [metrics.get("operating_margin"), metrics.get("net_margin"), metrics.get("roe")]
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return clamp_score(50 + sum(numeric) / len(numeric))


def _score_stability(metrics: dict[str, Any]) -> float | None:
    debt_ratio = metrics.get("debt_ratio")
    if not isinstance(debt_ratio, (int, float)):
        return None
    return clamp_score(80 - max(0, debt_ratio - 100) * 0.2)


def _volume_view(volume_ratio: Any) -> str:
    if not isinstance(volume_ratio, (int, float)):
        return "거래량 비율 계산 불가"
    if volume_ratio >= 5:
        return "평균 대비 거래량 급증, 과열 여부 확인 필요"
    if volume_ratio >= 1.5:
        return "평균 대비 거래량 증가"
    if volume_ratio < 0.7:
        return "평균 대비 거래량 부족"
    return "평균 수준 거래량"


def _risk_profile(risk_level: str) -> dict[str, Any]:
    return RISK_PROFILES.get(risk_level, RISK_PROFILES["medium"])
