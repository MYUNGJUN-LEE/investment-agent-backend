from typing import Any


POSITIVE_KEYWORDS = [
    "수주", "계약", "공급", "자사주", "실적 개선", "흑자전환", "증설",
    "승인", "수출", "MOU", "협력", "AI", "HBM", "데이터센터", "원전",
    "방산", "전력망", "변압기",
]

NEGATIVE_KEYWORDS = [
    "유상증자", "전환사채", "CB", "BW", "횡령", "배임", "거래정지",
    "관리종목", "감사의견", "계약 해지", "실적 부진", "적자전환",
    "소송", "제재", "리콜",
]


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, round(value, 1)))


def classify_text_impact(text: str) -> dict[str, Any]:
    normalized = text.lower()
    positive_hits = [kw for kw in POSITIVE_KEYWORDS if kw.lower() in normalized]
    negative_hits = [kw for kw in NEGATIVE_KEYWORDS if kw.lower() in normalized]

    if negative_hits and not positive_hits:
        return {
            "direction": "negative",
            "strength": min(100, 50 + len(negative_hits) * 10),
            "matched_keywords": negative_hits,
        }

    if positive_hits and not negative_hits:
        return {
            "direction": "positive",
            "strength": min(100, 45 + len(positive_hits) * 10),
            "matched_keywords": positive_hits,
        }

    if positive_hits and negative_hits:
        return {
            "direction": "uncertain",
            "strength": 55,
            "matched_keywords": positive_hits + negative_hits,
        }

    return {
        "direction": "uncertain",
        "strength": 30,
        "matched_keywords": [],
    }


def score_research_events(events: list[dict[str, Any]]) -> float:
    if not events:
        return 45.0

    score = 50.0
    for event in events:
        title = event.get("title", "") or ""
        summary = event.get("summary", "") or ""
        impact = classify_text_impact(f"{title} {summary}")

        if impact["direction"] == "positive":
            score += impact["strength"] * 0.12
        elif impact["direction"] == "negative":
            score -= impact["strength"] * 0.16

        if event.get("source") == "OpenDART":
            score += 3

    return clamp_score(score)


def score_financial_data(financial_data: dict[str, Any]) -> float:
    if not financial_data or financial_data.get("status", "").startswith("not_"):
        return 50.0
    if financial_data.get("status") in ("missing_api_key_or_corp_code", "missing_opendart_api_key", "missing_corp_code"):
        return 50.0
    if "error" in financial_data:
        return 45.0

    return 55.0


def score_chart_flow(price_data: dict[str, Any]) -> float:
    if not price_data or price_data.get("status") == "not_connected_yet":
        return 50.0

    score = 50.0
    change_rate = price_data.get("change_rate")
    volume_ratio = price_data.get("volume_ratio")

    if isinstance(change_rate, (int, float)):
        if 0 < change_rate < 5:
            score += 10
        elif change_rate >= 8:
            score -= 10
        elif change_rate < -3:
            score -= 10

    if isinstance(volume_ratio, (int, float)):
        if volume_ratio >= 3:
            score += 15
        elif volume_ratio < 0.7:
            score -= 10

    return clamp_score(score)


def calculate_final_score(
    research_score: float,
    financial_score: float,
    chart_flow_score: float,
    risk_score: float,
) -> float:
    return clamp_score(
        research_score * 0.30
        + financial_score * 0.20
        + chart_flow_score * 0.30
        + (100 - risk_score) * 0.20
    )
