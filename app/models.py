from typing import Any, Literal
from pydantic import BaseModel, Field


Market = Literal["KR", "US"]
StrategyType = Literal["daytrade", "swing", "midterm"]
RiskLevel = Literal["low", "medium", "high"]
FinalGrade = Literal["공격", "중립", "관망", "회피"]


class PipelineRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 000660")
    name: str | None = Field(None, description="Company name")
    market: Market = "KR"
    strategy_type: StrategyType = "daytrade"
    lookback_hours: int = Field(24, ge=1, le=24 * 30)
    risk_level: RiskLevel = "medium"


class PipelineResponse(BaseModel):
    symbol: str
    name: str | None
    market: Market
    strategy_type: StrategyType

    final_grade: FinalGrade
    entry_signal: bool
    exit_signal: bool
    confidence: float
    summary: str
    disclaimer: str

    scores: dict[str, float]

    entry_conditions: list[str]
    avoid_conditions: list[str]
    stop_loss_candidates: list[str]
    take_profit_candidates: list[str]
    time_exit_rule: str

    research_result: dict[str, Any]
    financial_result: dict[str, Any]
    chart_flow_result: dict[str, Any]
    devils_advocate_result: dict[str, Any]
