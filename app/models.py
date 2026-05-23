from typing import Any, Literal
from pydantic import BaseModel, Field


Market = Literal["KR"]
StrategyType = Literal["daytrade", "swing", "midterm"]
RiskLevel = Literal["low", "medium", "high"]
FinalGrade = Literal["공격", "중립", "관심", "관망", "회피"]
PaperSignalType = Literal["entry", "exit"]
LiveOrderSide = Literal["buy", "sell"]
LiveOrderType = Literal["limit"]
OrderPreviewAction = Literal["auto", "entry", "exit"]
OrderPreviewStatus = Literal["pending", "blocked"]
OrderExecutionMode = Literal["paper"]
AutoTradeExecutionMode = Literal["paper", "live"]
AutoTradeSessionStatus = Literal["active", "stopping", "stopped", "error", "not_found"]
GptAutoTradeCommand = Literal["start", "stop", "status"]


class PipelineRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 000660")
    name: str | None = Field(None, description="Company name")
    market: Market = "KR"
    strategy_type: StrategyType = "daytrade"
    lookback_hours: int = Field(24, ge=1, le=24 * 30)
    risk_level: RiskLevel = "medium"


class MarketContextRunRequest(BaseModel):
    symbol: str | None = Field(None, description="Optional ticker using the context")
    sector: str | None = Field(None, description="Optional sector to highlight")
    payload: dict[str, Any] | None = Field(
        None,
        description="Optional pre-fetched market context payload for manual import or tests.",
    )
    persist: bool = True


class MarketContextRunResponse(BaseModel):
    status: str
    trade_date: str | None = None
    market_regime: str
    risk_on_score: float | None = None
    kospi: dict[str, Any] | None = None
    kosdaq: dict[str, Any] | None = None
    usdkrw: dict[str, Any] | None = None
    vix: dict[str, Any] | None = None
    rates: dict[str, Any] | None = None
    sector_relative_strength: dict[str, Any] | None = None
    selected_sector_relative_strength: dict[str, Any] | None = None
    snapshot_id: int | None = None
    data_quality: dict[str, Any] | None = None
    message: str | None = None
    data_needed: list[str] | None = None


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


class PaperRunRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 005930")
    name: str | None = Field(None, description="Company name")
    market: Market = "KR"
    strategy_type: StrategyType = "daytrade"
    risk_level: RiskLevel = "medium"
    signal_type: PaperSignalType = Field(..., description="entry or exit")
    price: float = Field(..., gt=0, description="Virtual fill price")
    quantity: int | None = Field(
        None,
        gt=0,
        description="Entry quantity. For exit, omit to close the full position.",
    )
    confidence: float = Field(0.5, ge=0, le=1)
    reason: str | None = None
    source: str = "manual"
    signal_time: str | None = None
    decision_price: float | None = Field(None, gt=0)
    order_price: float | None = Field(None, gt=0)
    fill_price: float | None = Field(None, gt=0)
    filled_quantity: int | None = Field(None, ge=0)
    signal_score: float | None = Field(None, ge=0, le=100)
    position_size: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, gt=0)
    take_profit: float | None = Field(None, gt=0)
    market_regime: str | None = None
    model_version: str = "rule_based_v1"
    sector: str | None = None
    account_equity: float | None = Field(None, gt=0)
    risk_per_trade: float | None = Field(None, gt=0, le=0.02)
    cash_available: float | None = Field(None, ge=0)
    expected_gross_edge_bps: float | None = Field(
        None,
        description="Expected gross edge before all trading costs, in basis points.",
    )
    expected_win_bps: float | None = Field(
        None,
        description="Expected upside in basis points, used for reward/risk.",
    )
    expected_loss_bps: float | None = Field(
        None,
        description="Expected downside or stop distance in basis points.",
    )
    expected_sharpe: float | None = Field(
        None,
        description="Expected Sharpe ratio for this setup or strategy.",
    )
    commission_rate: float | None = Field(None, ge=0)
    sell_tax_rate: float | None = Field(None, ge=0)
    spread_bps: float | None = Field(None, ge=0)
    slippage_bps: float | None = Field(None, ge=0)
    fx_spread_bps: float | None = Field(None, ge=0)
    leverage: float = Field(1.0, ge=1.0)
    margin_interest_rate: float | None = Field(None, ge=0)
    borrow_fee_rate: float | None = Field(None, ge=0)
    expected_holding_days: float = Field(1.0, ge=0)
    fill_probability: float | None = Field(None, ge=0, le=1)
    is_short: bool = False
    market_beta: float | None = None


class PaperRunResponse(BaseModel):
    status: str
    signal_id: int
    order_id: int
    order_status: str
    symbol: str
    side: str
    price: float
    quantity: int
    amount: float
    effective_price: float | None = None
    total_cost: float = 0
    realized_pnl: float = 0
    cost_breakdown: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None
    message: str
    position: dict[str, Any] | None


class LiveOrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 005930")
    market: Market = "KR"
    risk_level: RiskLevel = "medium"
    side: LiveOrderSide
    order_type: LiveOrderType = Field(
        "limit",
        description="Only limit orders are allowed. Market orders are rejected.",
    )
    price: float = Field(..., gt=0, description="Limit order price")
    quantity: int = Field(..., gt=0)
    confirm_token: str
    client_order_id: str | None = Field(
        None,
        description="Optional idempotency key. Reusing it blocks duplicate live orders.",
    )
    session_id: str | None = Field(
        None,
        description="Optional auto-trading session id for order-state tracing.",
    )
    reason: str | None = None
    signal_time: str | None = None
    decision_price: float = Field(
        ...,
        gt=0,
        description="Reference price used to reject abnormal live order price deviation.",
    )
    order_price: float | None = Field(None, gt=0)
    signal_score: float | None = Field(None, ge=0, le=100)
    position_size: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, gt=0)
    take_profit: float | None = Field(None, gt=0)
    market_regime: str | None = None
    model_version: str = "rule_based_v1"
    sector: str | None = None
    account_equity: float | None = Field(None, gt=0)
    risk_per_trade: float | None = Field(None, gt=0, le=0.02)
    cash_available: float | None = Field(None, ge=0)
    expected_gross_edge_bps: float | None = None
    expected_win_bps: float | None = None
    expected_loss_bps: float | None = None
    expected_sharpe: float | None = None
    commission_rate: float | None = Field(None, ge=0)
    sell_tax_rate: float | None = Field(None, ge=0)
    spread_bps: float | None = Field(None, ge=0)
    slippage_bps: float | None = Field(None, ge=0)
    fx_spread_bps: float | None = Field(None, ge=0)
    leverage: float = Field(1.0, ge=1.0)
    margin_interest_rate: float | None = Field(None, ge=0)
    borrow_fee_rate: float | None = Field(None, ge=0)
    expected_holding_days: float = Field(1.0, ge=0)
    fill_probability: float | None = Field(None, ge=0, le=1)
    is_short: bool = False
    market_beta: float | None = None


class LiveOrderResponse(BaseModel):
    status: str
    message: str
    symbol: str
    side: LiveOrderSide
    order_type: LiveOrderType
    price: float
    quantity: int
    kis_result: dict[str, Any] | None = None
    broker_sync: dict[str, Any] | None = None
    order_state: dict[str, Any] | None = None


class OrderPreviewRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 005930")
    name: str | None = Field(None, description="Company name")
    market: Market = "KR"
    strategy_type: StrategyType = "daytrade"
    lookback_hours: int = Field(24, ge=1, le=24 * 30)
    risk_level: RiskLevel = "medium"
    requested_action: OrderPreviewAction = "auto"
    price: float = Field(..., gt=0, description="Limit price used for preview")
    quantity: int | None = Field(
        None,
        gt=0,
        description=(
            "Requested quantity. If omitted for an entry preview, it is "
            "recommended from account_equity, risk_per_trade, and stop_loss."
        ),
    )
    signal_time: str | None = None
    decision_price: float | None = Field(None, gt=0)
    order_price: float | None = Field(None, gt=0)
    signal_score: float | None = Field(None, ge=0, le=100)
    position_size: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, gt=0)
    take_profit: float | None = Field(None, gt=0)
    market_regime: str | None = None
    model_version: str = "rule_based_v1"
    sector: str | None = None
    account_equity: float | None = Field(None, gt=0)
    risk_per_trade: float | None = Field(None, gt=0, le=0.02)
    cash_available: float | None = Field(None, ge=0)
    expected_gross_edge_bps: float | None = None
    expected_win_bps: float | None = None
    expected_loss_bps: float | None = None
    expected_sharpe: float | None = None
    commission_rate: float | None = Field(None, ge=0)
    sell_tax_rate: float | None = Field(None, ge=0)
    spread_bps: float | None = Field(None, ge=0)
    slippage_bps: float | None = Field(None, ge=0)
    fx_spread_bps: float | None = Field(None, ge=0)
    leverage: float = Field(1.0, ge=1.0)
    margin_interest_rate: float | None = Field(None, ge=0)
    borrow_fee_rate: float | None = Field(None, ge=0)
    expected_holding_days: float = Field(1.0, ge=0)
    fill_probability: float | None = Field(None, ge=0, le=1)
    is_short: bool = False
    market_beta: float | None = None


class OrderPreviewResponse(BaseModel):
    status: OrderPreviewStatus
    preview_id: int
    preview_token: str | None
    symbol: str
    signal_type: PaperSignalType | None
    side: str | None
    price: float
    quantity: int
    amount: float
    recommended_quantity: dict[str, Any] | None = None
    message: str
    strategy_decision: dict[str, Any]
    risk_decision: dict[str, Any] | None
    cost_edge_decision: dict[str, Any] | None = None


class OrderConfirmRequest(BaseModel):
    preview_id: int
    preview_token: str
    execution_mode: OrderExecutionMode = "paper"


class OrderConfirmResponse(BaseModel):
    status: str
    preview_id: int
    execution_mode: OrderExecutionMode
    paper_result: PaperRunResponse


class AutoTradeSymbolConfig(BaseModel):
    symbol: str = Field(..., description="Stock ticker code, e.g. 005930")
    name: str | None = Field(None, description="Company name")
    market: Market = "KR"
    strategy_type: StrategyType = "daytrade"
    lookback_hours: int = Field(24, ge=1, le=24 * 30)
    risk_level: RiskLevel = "medium"
    requested_action: OrderPreviewAction = "auto"
    price: float | None = Field(
        None,
        gt=0,
        description="Fallback limit price. If omitted, the loop tries KIS current price.",
    )
    quantity: int | None = Field(None, gt=0)
    decision_price: float | None = Field(
        None,
        gt=0,
        description="Reference price. Defaults to the resolved loop price.",
    )
    order_price: float | None = Field(None, gt=0)
    signal_score: float | None = Field(None, ge=0, le=100)
    position_size: float | None = Field(None, ge=0)
    stop_loss: float | None = Field(None, gt=0)
    take_profit: float | None = Field(None, gt=0)
    market_regime: str | None = None
    model_version: str = "rule_based_v1"
    sector: str | None = None
    account_equity: float | None = Field(None, gt=0)
    risk_per_trade: float | None = Field(None, gt=0, le=0.02)
    cash_available: float | None = Field(None, ge=0)
    expected_gross_edge_bps: float | None = None
    expected_win_bps: float | None = None
    expected_loss_bps: float | None = None
    expected_sharpe: float | None = None
    commission_rate: float | None = Field(None, ge=0)
    sell_tax_rate: float | None = Field(None, ge=0)
    spread_bps: float | None = Field(None, ge=0)
    slippage_bps: float | None = Field(None, ge=0)
    fx_spread_bps: float | None = Field(None, ge=0)
    leverage: float = Field(1.0, ge=1.0)
    margin_interest_rate: float | None = Field(None, ge=0)
    borrow_fee_rate: float | None = Field(None, ge=0)
    expected_holding_days: float = Field(1.0, ge=0)
    fill_probability: float | None = Field(None, ge=0, le=1)
    is_short: bool = False
    market_beta: float | None = None


class AutoTradeStartRequest(BaseModel):
    symbols: list[AutoTradeSymbolConfig] = Field(default_factory=list, max_length=20)
    auto_discover_symbols: bool = Field(
        True,
        description=(
            "When true, an empty symbols list starts the universe scanner and "
            "trades only the final discovered candidates."
        ),
    )
    universe_seed_symbols: list[str] = Field(
        default_factory=list,
        max_length=200,
        description="Optional seed universe. If omitted, defaults and env seeds are used.",
    )
    universe_candidate_limit: int = Field(
        20,
        ge=1,
        le=50,
        description="Rule scanner candidate count before news/disclosure enrichment.",
    )
    universe_final_limit: int = Field(
        10,
        ge=1,
        le=20,
        description="Final candidate count passed into the auto-trading analyzer.",
    )
    execution_mode: AutoTradeExecutionMode = "paper"
    interval_seconds: int = Field(
        60,
        ge=10,
        le=3600,
        description="Default is 60 seconds; each cycle refreshes news, disclosures, and KIS data.",
    )
    max_cycles: int | None = Field(
        None,
        ge=1,
        le=10_000,
        description="Optional cycle limit. Omit to keep running until stopped.",
    )
    run_immediately: bool = True
    auto_confirm_paper: bool = True
    account_equity: float = Field(
        10_000_000,
        gt=0,
        description="Default account equity used for automatic quantity sizing.",
    )
    risk_per_trade: float = Field(
        0.005,
        gt=0,
        le=0.02,
        description="Default risk fraction per trade used for automatic quantity sizing.",
    )
    cash_available: float | None = Field(
        None,
        ge=0,
        description="Optional cash cap used for automatic quantity sizing.",
    )
    live_confirm_token: str | None = Field(
        None,
        description="Required only when execution_mode is live.",
    )


class AutoTradeStartResponse(BaseModel):
    session_id: str
    status: AutoTradeSessionStatus
    execution_mode: AutoTradeExecutionMode
    interval_seconds: int
    max_cycles: int | None
    started_at: str
    message: str
    universe_scan: dict[str, Any] | None = None


class AutoTradeStatusResponse(BaseModel):
    session_id: str
    status: AutoTradeSessionStatus
    execution_mode: AutoTradeExecutionMode | None = None
    interval_seconds: int | None = None
    max_cycles: int | None = None
    cycle_count: int = 0
    started_at: str | None = None
    updated_at: str | None = None
    next_run_at: str | None = None
    last_error: str | None = None
    last_results: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None


class AutoTradeStopResponse(BaseModel):
    session_id: str
    status: AutoTradeSessionStatus
    message: str


class AutoTradeRestartResponse(BaseModel):
    session_id: str
    status: AutoTradeSessionStatus
    message: str


class AutoTradeSessionsResponse(BaseModel):
    count: int
    sessions: list[AutoTradeStatusResponse]


class AutoTradeEvent(BaseModel):
    id: int
    session_id: str
    created_at: str
    event_type: str
    status: str
    message: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)


class AutoTradeEventsResponse(BaseModel):
    session_id: str
    count: int
    events: list[AutoTradeEvent]


class AutoTradeRunOnceResponse(BaseModel):
    execution_mode: AutoTradeExecutionMode
    results: list[dict[str, Any]]


class GptAutoTradeControlRequest(BaseModel):
    command: GptAutoTradeCommand = Field(
        ...,
        description="Use start to turn on auto-trading, stop to stop all active sessions, status to inspect it.",
    )
    execution_mode: AutoTradeExecutionMode = "paper"
    interval_seconds: int = Field(60, ge=10, le=3600)
    max_cycles: int | None = Field(None, ge=1, le=10_000)
    auto_discover_symbols: bool = True
    universe_seed_symbols: list[str] = Field(default_factory=list, max_length=200)
    universe_candidate_limit: int = Field(20, ge=1, le=50)
    universe_final_limit: int = Field(10, ge=1, le=20)
    auto_confirm_paper: bool = True
    account_equity: float = Field(
        10_000_000,
        gt=0,
        description="Default account equity used for automatic quantity sizing.",
    )
    risk_per_trade: float = Field(
        0.005,
        gt=0,
        le=0.02,
        description="Default risk fraction per trade used for automatic quantity sizing.",
    )
    cash_available: float | None = Field(
        None,
        ge=0,
        description="Optional cash cap used for automatic quantity sizing.",
    )
    force_new: bool = Field(
        False,
        description="When false, start reuses an already active auto-trading session.",
    )
    live_confirm_token: str | None = Field(
        None,
        description="Required only when execution_mode is live.",
    )


class GptAutoTradeControlResponse(BaseModel):
    status: str
    command: GptAutoTradeCommand | str
    message: str
    error_type: str | None = None
    http_status: int | None = None
    detail: Any | None = None
    started_session: AutoTradeStartResponse | None = None
    stopped_sessions: list[AutoTradeStopResponse] = Field(default_factory=list)
    active_sessions: list[AutoTradeStatusResponse] = Field(default_factory=list)
    recent_sessions: list[AutoTradeStatusResponse] = Field(default_factory=list)
    worker_status: dict[str, Any] | None = None
