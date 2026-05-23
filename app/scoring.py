from typing import Any


POSITIVE_KEYWORDS = [
    "수주", "대규모 수주", "신규 수주", "계약", "공급", "공급계약", "장기공급",
    "독점 공급", "납품", "수출", "수출 확대", "수출 허가", "MOU", "협력",
    "전략적 제휴", "파트너십", "인수", "합병", "지분 투자", "투자 유치",
    "자사주", "자사주 매입", "자사주 소각", "배당 확대", "무상증자",
    "실적 개선", "실적 호조", "실적 서프라이즈", "어닝 서프라이즈", "흑자전환",
    "턴어라운드", "가이던스 상향", "목표가 상향", "매출 성장", "영업이익 증가",
    "증설", "CAPEX", "신공장", "양산", "수율 개선", "가격 인상", "ASP 상승",
    "업황 회복", "승인", "FDA 승인", "품목허가", "임상 성공", "국책과제",
    "정책 수혜", "AI", "HBM", "CXL", "데이터센터", "온디바이스 AI",
    "반도체", "전력반도체", "로봇", "자율주행", "전장", "2차전지", "전고체",
    "리튬", "니켈", "구리", "희토류", "우라늄", "원전", "SMR", "전력망",
    "변압기", "전력기기", "방산", "우주항공", "조선", "LNG", "해운 운임",
    "바이오시밀러", "CDMO", "환율 수혜",
]

NEGATIVE_KEYWORDS = [
    "유상증자", "전환사채", "CB", "BW", "신주인수권", "감자", "오버행",
    "보호예수 해제", "블록딜", "대주주 매도", "횡령", "배임", "압수수색",
    "수사", "기소", "소송", "제재", "과징금", "벌금", "거래정지", "관리종목",
    "상장폐지", "불성실공시", "감사의견", "감사의견 거절", "한정", "계속기업",
    "계약 해지", "공급 중단", "납품 지연", "실적 부진", "실적 쇼크", "어닝 쇼크",
    "적자전환", "매출 감소", "영업손실", "가이던스 하향", "목표가 하향",
    "리콜", "품질 이슈", "화재", "침수", "생산 중단", "파업", "규제",
    "금리 상승", "환율 부담", "원가 상승", "투자주의", "투자경고",
    "주가급등 조회공시", "단기과열", "매매거래정지",
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
    metrics = financial_data.get("metrics") or {}
    growth = financial_data.get("growth_metrics") or {}
    valuation = financial_data.get("valuation_metrics") or {}
    if not metrics:
        return 50.0

    score = 50.0
    operating_margin = metrics.get("operating_margin")
    net_margin = metrics.get("net_margin")
    roe = metrics.get("roe")
    debt_ratio = metrics.get("debt_ratio")
    revenue_growth = growth.get("revenue_growth")
    operating_income_growth = growth.get("operating_income_growth")

    if isinstance(operating_margin, (int, float)):
        if operating_margin >= 15:
            score += 10
        elif operating_margin >= 5:
            score += 5
        elif operating_margin < 0:
            score -= 12

    if isinstance(net_margin, (int, float)):
        if net_margin >= 10:
            score += 8
        elif net_margin < 0:
            score -= 10

    if isinstance(roe, (int, float)):
        if roe >= 15:
            score += 10
        elif roe >= 8:
            score += 5
        elif roe < 0:
            score -= 10

    if isinstance(debt_ratio, (int, float)):
        if debt_ratio <= 100:
            score += 8
        elif debt_ratio >= 250:
            score -= 12

    if isinstance(revenue_growth, (int, float)):
        if revenue_growth >= 15:
            score += 8
        elif revenue_growth < -10:
            score -= 8

    if isinstance(operating_income_growth, (int, float)):
        if operating_income_growth >= 20:
            score += 8
        elif operating_income_growth < -20:
            score -= 8

    per = valuation.get("per")
    pbr = valuation.get("pbr")
    ev_ebitda = valuation.get("ev_ebitda")
    if isinstance(per, (int, float)):
        if 0 < per <= 12:
            score += 4
        elif per >= 40:
            score -= 4
    if isinstance(pbr, (int, float)):
        if 0 < pbr <= 1.5:
            score += 3
        elif pbr >= 5:
            score -= 3
    if isinstance(ev_ebitda, (int, float)):
        if 0 < ev_ebitda <= 8:
            score += 3
        elif ev_ebitda >= 20:
            score -= 3

    return clamp_score(score)


def score_chart_flow(price_data: dict[str, Any]) -> float:
    if not price_data or price_data.get("status") != "connected":
        return 50.0

    score = 50.0
    change_rate = price_data.get("change_rate")
    volume_ratio = price_data.get("volume_ratio")
    trend = price_data.get("trend")
    price_position = price_data.get("price_position") or {}
    intraday = price_data.get("intraday") or {}
    technical = price_data.get("latest_technical_features") or {}

    if isinstance(change_rate, (int, float)):
        if 0 < change_rate < 5:
            score += 10
        elif change_rate >= 8:
            score -= 10
        elif change_rate < -3:
            score -= 10

    if isinstance(volume_ratio, (int, float)):
        if 1.5 <= volume_ratio < 5:
            score += 15
        elif volume_ratio >= 5:
            score += 5
        elif volume_ratio < 0.7:
            score -= 10

    if trend == "uptrend":
        score += 10
    elif trend == "downtrend":
        score -= 12

    if price_position.get("above_ma5"):
        score += 5
    if price_position.get("above_ma20"):
        score += 8
    if price_data.get("overheated"):
        score -= 10

    intraday_score = intraday.get("intraday_score")
    if isinstance(intraday_score, (int, float)):
        score += (intraday_score - 50) * 0.35

    execution_strength = intraday.get("execution_strength")
    if isinstance(execution_strength, (int, float)):
        if execution_strength >= 140:
            score += 8
        elif execution_strength >= 120:
            score += 5
        elif execution_strength < 80:
            score -= 8

    orderbook_imbalance = intraday.get("orderbook_imbalance")
    if isinstance(orderbook_imbalance, (int, float)):
        if orderbook_imbalance >= 0.2:
            score += 6
        elif orderbook_imbalance <= -0.2:
            score -= 7

    spread_pct = intraday.get("spread_pct")
    if isinstance(spread_pct, (int, float)) and spread_pct > 0.6:
        score -= 8

    if intraday.get("above_minute_vwap") is True:
        score += 5
    elif intraday.get("above_minute_vwap") is False:
        score -= 6

    minute_momentum = intraday.get("minute_momentum_pct")
    if isinstance(minute_momentum, (int, float)):
        if 0.2 <= minute_momentum <= 2.5:
            score += 5
        elif minute_momentum < -1:
            score -= 6

    smart_money = intraday.get("smart_money_net_buy_5d")
    if isinstance(smart_money, (int, float)):
        if smart_money > 0:
            score += 5
        elif smart_money < 0:
            score -= 5

    if technical.get("high_breakout_20d"):
        score += 6
    if technical.get("low_breakdown_20d"):
        score -= 8
    if isinstance(technical.get("ma20_slope"), (int, float)):
        score += 4 if technical["ma20_slope"] > 0 else -4
    if isinstance(technical.get("macd_histogram"), (int, float)):
        score += 4 if technical["macd_histogram"] > 0 else -4
    if isinstance(technical.get("adx_14"), (int, float)) and technical["adx_14"] >= 25:
        score += 3
    if isinstance(technical.get("atr_14_pct"), (int, float)) and technical["atr_14_pct"] > 0.08:
        score -= 5

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
