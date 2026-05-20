from __future__ import annotations

from app.models import PipelineRequest
from app.services.naver_news import search_naver_news

from app.agents import (
    run_research_agent,
    run_financial_agent,
    run_chart_flow_agent,
    run_devils_advocate_agent,
    run_final_check_agent,
)


DISCLAIMER = (
    "본 결과는 공개 데이터 기반의 분석 보조 자료이며, 확정적 매수/매도 지시나 수익 보장이 아닙니다. "
    "실제 매매 전 최신 시세, 공시, 뉴스, 수급, 본인의 손실 감내 범위를 반드시 확인하세요."
)


async def run_full_pipeline(request):
    ticker = request.ticker
    company_name = request.company_name if hasattr(request, "company_name") else ticker

    naver_news = await search_naver_news(
        query=company_name,
        display=10
        sort="date"
    )

    # Research Agent에 뉴스 데이터 전달
    research_result = {
        "news_source": "Naver Search API",
        "naver_connected": naver_news.get("connected"),
        "news_items": naver_news.get("items", []),
    }

    # 이후 기존 5개 agent 분석에 naver_news를 포함
    financial_result = ...
    chart_result = ...
    devil_result = ...
    final_result = ...

    return {
        "ticker": ticker,
        "company_name": company_name,
        "naver_news": naver_news["items"],
        "research_result": research_result,
        "financial_result": financial_result,
        "chart_result": chart_result,
        "devil_result": devil_result,
        "final_result": final_result
    }

def run_full_pipeline(req: PipelineRequest) -> dict:
    naver_news_result = search_naver_news(req)
    research_result = run_research_agent(req)
    if isinstance(research_result, dict):
        research_result["naver_news"] = naver_news_result
        research_result["naver_connected"] = naver_news_result.get("connected")
        research_result["news_source"] = "Naver Search API"
    else:
        research_result = {
            "research_agent_result": research_result,
            "naver_news": naver_news_result,
            "naver_connected": naver_news_result.get("connected"),
            "news_source": "Naver Search API",
        }

    financial_result = run_financial_agent(req, research_result)
    chart_flow_result = run_chart_flow_agent(req, research_result, financial_result)
    devils_result = run_devils_advocate_agent(
        req,
        research_result,
        financial_result,
        chart_flow_result,
    )
    final_result = run_final_check_agent(
        req,
        research_result,
        financial_result,
        chart_flow_result,
        devils_result,
    )

    return {
        "symbol": req.symbol,
        "name": req.name,
        "company_name": req.company_name if hasattr(req, "company_name") else req.symbol,
        "market": req.market,
        "strategy_type": req.strategy_type,
        "final_grade": final_result["final_grade"],
        "entry_signal": final_result["entry_signal"],
        "exit_signal": final_result["exit_signal"],
        "confidence": final_result["confidence"],
        "summary": final_result["summary"],
        "disclaimer": DISCLAIMER,
        "scores": final_result["scores"],
        "entry_conditions": final_result["entry_conditions"],
        "avoid_conditions": final_result["avoid_conditions"],
        "stop_loss_candidates": final_result["stop_loss_candidates"],
        "take_profit_candidates": final_result["take_profit_candidates"],
        "time_exit_rule": final_result["time_exit_rule"],
        "research_result": research_result,
        "financial_result": financial_result,
        "chart_flow_result": chart_flow_result,
        "devils_advocate_result": devils_result,
    }
