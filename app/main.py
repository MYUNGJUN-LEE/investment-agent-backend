from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from app.services.naver_news import search_naver_news

from app.brokers.kis_client import KisClient
from app.config import settings
from app.models import (
    AutoTradeEventsResponse,
    AutoTradeRunOnceResponse,
    AutoTradeRestartResponse,
    AutoTradeSessionsResponse,
    AutoTradeStartRequest,
    AutoTradeStartResponse,
    AutoTradeStatusResponse,
    AutoTradeStopResponse,
    GptAutoTradeControlRequest,
    GptAutoTradeControlResponse,
    LiveOrderRequest,
    LiveOrderResponse,
    MarketContextRunRequest,
    MarketContextRunResponse,
    OrderConfirmRequest,
    OrderConfirmResponse,
    OrderPreviewRequest,
    OrderPreviewResponse,
    PaperRunRequest,
    PaperRunResponse,
    PipelineRequest,
    PipelineResponse,
)
from app.data_sources.market_context import StaticMarketContextProvider, fetch_market_context
from app.services.pipeline import run_full_pipeline
from app.trading.auto_trading import (
    AutoTradingError,
    control_auto_trading_from_gpt,
    get_auto_trading_status,
    list_auto_trading_events,
    list_auto_trading_sessions,
    restart_auto_trading,
    run_auto_trading_once,
    start_auto_trading,
    stop_auto_trading,
)
from app.trading.broker_sync import sync_kis_account
from app.trading.kis_paper_e2e import KisPaperE2EError, preflight_kis_paper_e2e
from app.trading.live_trading import LiveTradingError, execute_live_order
from app.trading.market_monitor import (
    get_monitor_status,
    process_due_monitor_jobs,
    run_monitor_job,
)
from app.trading.order_approval import (
    OrderApprovalError,
    confirm_order_preview,
    create_order_preview,
)
from app.trading.paper_trading import run_paper_once
from app.trading.universe_scanner import (
    get_latest_universe_scan,
    scan_universe_for_auto_trade,
)


def _cors_allow_origins() -> list[str]:
    return [
        origin.strip()
        for origin in settings.cors_allow_origins.split(",")
        if origin.strip()
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.workers.manager import start_on_app_startup_if_enabled

    start_on_app_startup_if_enabled()
    yield


app = FastAPI(
    title="Investment Agent API",
    version="1.0.0",
    description="Backend API for Custom GPT investment analysis pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if _is_gpt_action_path(request):
        return JSONResponse(
            status_code=200,
            content=_gpt_error_payload(
                error_type="http_error",
                http_status=exc.status_code,
                message=str(exc.detail),
                detail=exc.detail,
            ),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    if _is_gpt_action_path(request):
        return JSONResponse(
            status_code=200,
            content=_gpt_error_payload(
                error_type="validation_error",
                http_status=422,
                message="Invalid Custom GPT action request body or parameters.",
                detail=exc.errors(),
            ),
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _is_gpt_action_path(request: Request) -> bool:
    return request.url.path.rstrip("/").startswith("/gpt/")


def _gpt_error_payload(
    *,
    error_type: str,
    http_status: int,
    message: str,
    detail,
) -> dict:
    return {
        "status": "error",
        "command": "unknown",
        "message": message,
        "error_type": error_type,
        "http_status": http_status,
        "detail": detail,
        "started_session": None,
        "stopped_sessions": [],
        "active_sessions": [],
        "recent_sessions": [],
        "worker_status": None,
    }


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    """
    If BACKEND_API_KEY is set in .env, every request must include:
    X-API-Key: <BACKEND_API_KEY>

    Custom GPT Actions should be configured with X-API-Key, but accepting a
    bearer token as a fallback makes production diagnosis less brittle when the
    action auth type is accidentally set to Bearer.
    """
    if settings.backend_api_key:
        candidates = [x_api_key]
        if authorization:
            scheme, _, token = authorization.partition(" ")
            candidates.append(token if scheme.lower() == "bearer" and token else authorization)
        if settings.backend_api_key not in candidates:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Send X-API-Key or Authorization: Bearer.",
            )

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "message": "Investment Agent Backend is running",
        "health": "/health",
        "docs": "/docs"
    }

@app.get("/health")
@app.get("/health/", include_in_schema=False)
def health_check():
    return _health_payload()


@app.head("/health", include_in_schema=False)
@app.head("/health/", include_in_schema=False)
def health_check_head():
    return Response(status_code=200)


@app.options("/health", include_in_schema=False)
@app.options("/health/", include_in_schema=False)
def health_check_options():
    return Response(
        status_code=204,
        headers={"Allow": "GET, HEAD, OPTIONS"},
    )


@app.post("/health", include_in_schema=False)
@app.post("/health/", include_in_schema=False)
def health_check_post():
    return _health_payload()


@app.get("/healthz", include_in_schema=False)
@app.get("/healthz/", include_in_schema=False)
def healthz_check():
    return _health_payload()


@app.get("/gpt/health")
@app.get("/gpt/health/", include_in_schema=False)
def gpt_health_check():
    return _health_payload()


@app.post("/gpt/health", include_in_schema=False)
@app.post("/gpt/health/", include_in_schema=False)
def gpt_health_check_post():
    return _health_payload()


@app.head("/gpt/health", include_in_schema=False)
@app.head("/gpt/health/", include_in_schema=False)
def gpt_health_check_head():
    return Response(status_code=200)


@app.options("/gpt/health", include_in_schema=False)
@app.options("/gpt/health/", include_in_schema=False)
def gpt_health_check_options():
    return Response(
        status_code=204,
        headers={"Allow": "GET, HEAD, OPTIONS"},
    )


@app.get("/action-schema.yaml", include_in_schema=False)
@app.get("/.well-known/openapi.yaml", include_in_schema=False)
def custom_gpt_action_schema():
    schema_path = Path(__file__).resolve().parents[1] / "action_schema.gpt-control.yaml"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Action schema not found")
    return PlainTextResponse(
        content=schema_path.read_text(encoding="utf-8"),
        media_type="text/yaml",
    )


def _health_payload():
    return {
        "status": "ok",
        "service": settings.app_name,
        "auth_enabled": bool(settings.backend_api_key),
    }

@app.get("/naver/news", dependencies=[Depends(verify_api_key)])
def get_naver_news(query: str, display: int = 10):
    return search_naver_news(query=query, display=display)


@app.post(
    "/market-context/run-once",
    response_model=MarketContextRunResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="runMarketContextOnce",
    summary="Fetch, calculate, and store daily market context",
)
def run_market_context_once(req: MarketContextRunRequest):
    provider = StaticMarketContextProvider(req.payload) if req.payload is not None else None
    return fetch_market_context(
        symbol=req.symbol,
        sector=req.sector,
        provider=provider,
        persist=req.persist,
    )


@app.head("/")
def root_head():
    return Response(status_code=200)
    
@app.post(
    "/run-full-pipeline",
    response_model=PipelineResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="runFullPipeline",
    summary="Run full investment analysis pipeline",
    description="Runs the full investment analysis pipeline for a requested stock.",
)
def run_pipeline(req: PipelineRequest):
    return run_full_pipeline(req)


@app.post(
    "/paper/run-once",
    response_model=PaperRunResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="runPaperTradingOnce",
    summary="Record one paper-trading signal and virtual fill",
)
def run_paper_trading_once(req: PaperRunRequest):
    return run_paper_once(req)


@app.post(
    "/orders/preview",
    response_model=OrderPreviewResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="previewOrder",
    summary="Create a strategy and risk checked paper-order preview",
)
def preview_order(req: OrderPreviewRequest):
    try:
        return create_order_preview(req)
    except OrderApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/orders/confirm",
    response_model=OrderConfirmResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="confirmOrderPreview",
    summary="Confirm a pending order preview and execute it in paper trading",
)
def confirm_order(req: OrderConfirmRequest):
    try:
        return confirm_order_preview(req)
    except OrderApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/live/orders",
    response_model=LiveOrderResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="submitLiveLimitOrder",
    summary="Submit a live KIS limit order when all live-trading gates pass",
)
def submit_live_order(req: LiveOrderRequest):
    try:
        return execute_live_order(req)
    except LiveTradingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/auto-trading/start",
    response_model=AutoTradeStartResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="startAutoTrading",
    summary="Start a background auto-trading loop",
)
def start_auto_trading_session(req: AutoTradeStartRequest):
    try:
        return start_auto_trading(req)
    except AutoTradingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/auto-trading/run-once",
    response_model=AutoTradeRunOnceResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="runAutoTradingOnce",
    summary="Run one auto-trading cycle synchronously",
)
def run_auto_trading_once_endpoint(req: AutoTradeStartRequest):
    try:
        return run_auto_trading_once(req)
    except AutoTradingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/gpt/auto-trading/control",
    response_model=GptAutoTradeControlResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="controlAutoTradingFromGpt",
    summary="Turn auto-trading on or off from Custom GPT",
)
def control_auto_trading_endpoint(req: GptAutoTradeControlRequest):
    try:
        return control_auto_trading_from_gpt(req)
    except AutoTradingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/universe/scan",
    dependencies=[Depends(verify_api_key)],
    operation_id="scanTradingUniverse",
    summary="Discover and rank auto-trading candidates",
)
def scan_trading_universe(req: AutoTradeStartRequest):
    return scan_universe_for_auto_trade(req)


@app.get(
    "/universe/latest",
    dependencies=[Depends(verify_api_key)],
    operation_id="getLatestUniverseScan",
    summary="Get the latest stored universe scanner result",
)
def latest_trading_universe():
    return get_latest_universe_scan()


@app.get(
    "/auto-trading/sessions",
    response_model=AutoTradeSessionsResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="listAutoTradingSessions",
    summary="List persistent auto-trading sessions",
)
def list_auto_trading_sessions_endpoint(
    status: str | None = None,
    limit: int = 50,
):
    return list_auto_trading_sessions(status=status, limit=limit)


@app.get(
    "/auto-trading/status/{session_id}",
    response_model=AutoTradeStatusResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="getAutoTradingStatus",
    summary="Get auto-trading session status",
)
def get_auto_trading_session_status(session_id: str):
    return get_auto_trading_status(session_id)


@app.get(
    "/auto-trading/events/{session_id}",
    response_model=AutoTradeEventsResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="listAutoTradingEvents",
    summary="List recent auto-trading events for a session",
)
def list_auto_trading_events_endpoint(session_id: str, limit: int = 100):
    return list_auto_trading_events(session_id=session_id, limit=limit)


@app.post(
    "/auto-trading/stop/{session_id}",
    response_model=AutoTradeStopResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="stopAutoTrading",
    summary="Stop a background auto-trading loop",
)
def stop_auto_trading_session(session_id: str):
    return stop_auto_trading(session_id)


@app.post(
    "/auto-trading/restart/{session_id}",
    response_model=AutoTradeRestartResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="restartAutoTrading",
    summary="Restart a stopped or errored background auto-trading loop",
)
def restart_auto_trading_session(session_id: str):
    try:
        return restart_auto_trading(session_id)
    except AutoTradingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post(
    "/broker/kis/sync",
    dependencies=[Depends(verify_api_key)],
    operation_id="syncKisBrokerState",
    summary="Synchronize KIS account balance, positions, and recent executions",
)
def sync_kis_broker_state():
    try:
        return sync_kis_account()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/broker/kis/paper-preflight",
    dependencies=[Depends(verify_api_key)],
    operation_id="preflightKisPaperE2E",
    summary="Validate KIS paper connectivity without placing an order",
)
def preflight_kis_paper_state(symbol: str = "005930"):
    try:
        return preflight_kis_paper_e2e(symbol=symbol)
    except KisPaperE2EError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get(
    "/broker/kis/config-status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getKisConfigStatus",
    summary="Inspect non-secret KIS runtime configuration",
)
def get_kis_config_status():
    client = KisClient()
    diagnostics = client.runtime_diagnostics()
    diagnostics.update(
        {
            "enable_live_trading": settings.enable_live_trading,
            "embedded_worker_broker_sync_enabled": (
                settings.embedded_worker_broker_sync_enabled
            ),
            "broker_sync_interval_seconds": settings.broker_sync_interval_seconds,
        }
    )
    return diagnostics


@app.get(
    "/monitor/status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getMarketMonitorStatus",
    summary="Get periodic market monitor job status",
)
def get_market_monitor_status():
    return get_monitor_status()


@app.get(
    "/workers/status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getEmbeddedWorkerStatus",
    summary="Get embedded worker status",
)
def get_embedded_worker_status():
    from app.workers.manager import embedded_worker_status

    return embedded_worker_status()


@app.post(
    "/monitor/run-due",
    dependencies=[Depends(verify_api_key)],
    operation_id="runDueMarketMonitorJobs",
    summary="Run currently due 1m, 5m, and 10m market monitor jobs",
)
def run_due_market_monitor_jobs():
    return {"processed": process_due_monitor_jobs()}


@app.post(
    "/monitor/run/{job_name}",
    dependencies=[Depends(verify_api_key)],
    operation_id="runMarketMonitorJob",
    summary="Run one market monitor job immediately",
)
def run_market_monitor_job(job_name: str):
    try:
        return run_monitor_job(job_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/dashboard/auto-trading",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def auto_trading_dashboard():
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auto Trading Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --text: #18202a;
      --muted: #627084;
      --line: #d9dee7;
      --panel: #ffffff;
      --ok: #0f7b57;
      --warn: #9a6200;
      --bad: #b42318;
      --accent: #155eef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    main { padding: 24px; display: grid; gap: 18px; }
    .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    input, select, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.danger { color: var(--bad); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 650; background: #f0f3f8; }
    tr:last-child td { border-bottom: 0; }
    .status { font-weight: 700; }
    .active { color: var(--ok); }
    .stopped { color: var(--muted); }
    .error { color: var(--bad); }
    .events {
      white-space: pre-wrap;
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      min-height: 120px;
      overflow: auto;
    }
    .muted { color: var(--muted); }
    @media (max-width: 820px) {
      header, main { padding: 16px; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Auto Trading Dashboard</h1>
    <div class="toolbar">
      <input id="apiKey" type="password" placeholder="X-API-Key" />
      <select id="statusFilter">
        <option value="">All</option>
        <option value="active">Active</option>
        <option value="stopped">Stopped</option>
        <option value="error">Error</option>
      </select>
      <button class="primary" onclick="loadSessions()">Refresh</button>
    </div>
  </header>
  <main>
    <div class="muted" id="summary">Loading sessions...</div>
    <table>
      <thead>
        <tr>
          <th>Session</th>
          <th>Status</th>
          <th>Mode</th>
          <th>Cycle</th>
          <th>Next Run</th>
          <th>Updated</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="sessions"></tbody>
    </table>
    <section>
      <h2 style="font-size:16px;margin:0 0 8px;">Recent Events</h2>
      <div class="events" id="events">Select a session.</div>
    </section>
  </main>
  <script>
    const apiKey = document.getElementById("apiKey");
    const statusFilter = document.getElementById("statusFilter");
    const sessionsBody = document.getElementById("sessions");
    const summary = document.getElementById("summary");
    const eventsBox = document.getElementById("events");
    apiKey.value = sessionStorage.getItem("autoTradingApiKey") || "";
    apiKey.addEventListener("change", () => sessionStorage.setItem("autoTradingApiKey", apiKey.value));

    function headers() {
      const value = apiKey.value.trim();
      return value ? {"X-API-Key": value} : {};
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {...options, headers: {...headers(), ...(options.headers || {})}});
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      return response.json();
    }

    async function loadSessions() {
      try {
        const filter = statusFilter.value ? `?status=${encodeURIComponent(statusFilter.value)}` : "";
        const data = await api(`/auto-trading/sessions${filter}`);
        summary.textContent = `${data.count} session(s), refreshed ${new Date().toLocaleTimeString()}`;
        sessionsBody.innerHTML = data.sessions.map(row => `
          <tr>
            <td><code>${row.session_id}</code></td>
            <td class="status ${row.status}">${row.status}</td>
            <td>${row.execution_mode || ""}</td>
            <td>${row.cycle_count || 0}</td>
            <td>${row.next_run_at || ""}</td>
            <td>${row.updated_at || ""}</td>
            <td>
              <button onclick="loadEvents('${row.session_id}')">Events</button>
              <button class="danger" onclick="stopSession('${row.session_id}')" ${row.status !== "active" ? "disabled" : ""}>Stop</button>
              <button onclick="restartSession('${row.session_id}')">Restart</button>
            </td>
          </tr>
        `).join("");
      } catch (err) {
        summary.textContent = `Failed to load sessions: ${err.message}`;
      }
    }

    async function loadEvents(sessionId) {
      try {
        const data = await api(`/auto-trading/events/${sessionId}?limit=30`);
        eventsBox.textContent = JSON.stringify(data.events, null, 2);
      } catch (err) {
        eventsBox.textContent = `Failed to load events: ${err.message}`;
      }
    }

    async function stopSession(sessionId) {
      await api(`/auto-trading/stop/${sessionId}`, {method: "POST"});
      await loadSessions();
      await loadEvents(sessionId);
    }

    async function restartSession(sessionId) {
      await api(`/auto-trading/restart/${sessionId}`, {method: "POST"});
      await loadSessions();
      await loadEvents(sessionId);
    }

    statusFilter.addEventListener("change", loadSessions);
    loadSessions();
    setInterval(loadSessions, 5000);
  </script>
</body>
</html>
        """
    )


@app.post("/runMorningBriefing")
def run_morning_briefing(x_api_key: str | None = Header(default=None)):
    if settings.backend_api_key:
        if x_api_key != settings.backend_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    req = PipelineRequest(
        symbol="005930",
        name="삼성전자",
        market="KR",
        strategy_type="swing",
        lookback_hours=72,
        risk_level="medium",
    )
    result = run_full_pipeline(req)

    return {
        "status": "success",
        "message": "Morning briefing completed",
        "result": result
    }
