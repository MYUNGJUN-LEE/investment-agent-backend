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

    data_needed = []
    if financial_data.get("status") in ("missing_opendart_api_key", "missing_corp_code"):
        data_needed.append("OpenDART 재무제표 조회를 위한 API 키와 corp_code 매핑")
    if financial_data.get("error"):
        data_needed.append("OpenDART 재무 API 오류 확인")

    return {
        "financial_score": score,
        "growth_score": None,
        "profitability_score": None,
        "stability_score": None,
        "valuation_score": None,
        "summary": "재무 데이터는 현재 raw OpenDART 기반으로만 반환됩니다. 핵심 계정 파싱은 다음 단계에서 추가하세요.",
        "strengths": [],
        "weaknesses": [],
        "valuation_view": "unknown",
        "financial_risks": [
            "실시간 컨센서스, PER/PBR, 동종업계 비교는 아직 연결되지 않았습니다."
        ],
        "data_needed": data_needed,
        "raw_financial_data_preview": _preview(financial_data),
    }


def run_chart_flow_agent(
    req: PipelineRequest,
    research_result: dict[str, Any],
    financial_result: dict[str, Any],
) -> dict[str, Any]:
    price_data = fetch_price_data(req.symbol)
    score = score_chart_flow(price_data)

    data_needed = []
    if price_data.get("status") in ("not_connected_yet", "not_implemented_yet"):
        data_needed.append("KIS 또는 키움 시세 API 연결")
        data_needed.append("현재가, 분봉, 거래량, 거래대금, VWAP 데이터")

    return {
        "chart_flow_score": score,
        "trend": "unknown",
        "entry_signal": False,
        "exit_signal": False,
        "overheated": False,
        "support_levels": [],
        "resistance_levels": [],
        "volume_view": "시세 API 미연결로 거래량 판단 불가",
        "flow_view": "외국인/기관 수급 미연결",
        "entry_conditions": [
            "전고점 돌파",
            "5분 거래대금 증가",
            "VWAP 위 유지",
            "섹터 대장주 동반 상승",
        ],
        "avoid_conditions": [
            "거래량 없는 상승",
            "전고점 돌파 실패",
            "지수 급락",
            "급등 후 장대음봉",
        ],
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

    research_score = research_result.get("research_score", 50)
    if research_score < 45:
        risk_score += 10

    return {
        "risk_score": clamp_score(risk_score),
        "main_weakness": "현재 버전은 공시 중심이며, 실시간 시세/수급/뉴스 연결이 완전하지 않아 진입 타이밍 판단은 제한적입니다.",
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

    if final_score >= 75 and risk_score <= 50:
        final_grade = "공격"
        entry_signal = True
    elif final_score >= 65:
        final_grade = "중립"
        entry_signal = False
    elif final_score >= 50:
        final_grade = "관망"
        entry_signal = False
    else:
        final_grade = "회피"
        entry_signal = False

    if chart_flow_result.get("price_data", {}).get("status") in ("not_connected_yet", "not_implemented_yet"):
        entry_signal = False
        final_grade = "관망" if final_grade in ("공격", "중립") else final_grade

    return {
        "final_grade": final_grade,
        "entry_signal": entry_signal,
        "exit_signal": bool(chart_flow_result.get("exit_signal", False)),
        "confidence": round(final_score / 100, 2),
        "summary": f"최종 점수 {final_score}점. 현재 등급은 {final_grade}입니다. 실시간 시세/수급 연결 전에는 자동 진입 신호를 제한합니다.",
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


def _preview(data: Any, limit: int = 5) -> Any:
    if isinstance(data, dict) and isinstance(data.get("list"), list):
        preview = dict(data)
        preview["list"] = data["list"][:limit]
        preview["list_truncated"] = len(data["list"]) > limit
        return preview
    return data
