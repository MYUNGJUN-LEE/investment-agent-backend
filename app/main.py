from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import re
import sqlite3
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from app.services.naver_news import search_naver_news

from app.brokers.kis_client import KisApiError, KisClient
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
from app.maintenance.data_reset import RESET_CONFIRMATION, reset_trading_data
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
from app.trading.auto_tuning import (
    generate_auto_tuning_recommendations,
    latest_auto_tuning_recommendation,
)
from app.trading.broker_sync import sync_kis_account
from app.trading.edge_calibration import (
    calibrate_edge_model,
    edge_entry_gate,
    get_edge_calibration_status,
    get_edge_training_sample_summary,
    refresh_edge_training_samples,
    refresh_top10_performance_if_due,
)
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
    initialize_universe_db,
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


@app.middleware("http")
async def action_response_contract_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as exc:
        if _is_gpt_action_path(request):
            return _gpt_exception_response(exc)
        raise

    if _is_gpt_action_path(request) and response.status_code >= 300:
        return _gpt_json(
            _gpt_error_payload(
                error_type="http_error",
                http_status=response.status_code,
                message=(
                    "Backend action route returned a non-2xx response; converted "
                    "to JSON so GPT Actions can read the diagnostic payload."
                ),
                detail={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "location": response.headers.get("location"),
                },
            )
        )
    return response


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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if _is_gpt_action_path(request):
        return JSONResponse(
            status_code=200,
            content=_gpt_error_payload(
                error_type="server_error",
                http_status=500,
                message=str(exc) or "Unhandled backend error.",
                detail=str(exc),
            ),
        )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal Server Error"},
    )


def _is_gpt_action_path(request: Request) -> bool:
    path = request.url.path.rstrip("/")
    if path == "/gpt" or path.startswith("/gpt/"):
        return True
    if path.startswith((
        "/auto-trading/status/",
        "/auto-trading/events/",
        "/auto-trading/stop/",
        "/auto-trading/restart/",
    )):
        return True
    return path in {
        "/auto-trading",
        "/auto-trading/sessions",
        "/auto-trading/status",
        "/auto-trading/stop",
    }


def _gpt_error_payload(
    *,
    error_type: str,
    http_status: int,
    message: str,
    detail,
    command: str = "unknown",
) -> dict:
    payload = {
        "status": "error",
        "command": command,
        "message": message,
        "error_type": error_type,
        "http_status": http_status,
        "stopped_sessions": [],
        "active_sessions": [],
        "recent_sessions": [],
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


def _gpt_success_payload(message: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = {"status": "success", "message": message}
    content.update(_drop_none_values(payload))
    return content


def _gpt_json(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=200, content=_drop_none_values(payload))


def _gpt_exception_response(
    exc: Exception,
    *,
    command: str = "unknown",
    error_type: str = "server_error",
    http_status: int = 500,
) -> JSONResponse:
    if isinstance(exc, AutoTradingError):
        error_type = "auto_trading_error"
        http_status = exc.status_code
    return _gpt_json(
        _gpt_error_payload(
            error_type=error_type,
            http_status=http_status,
            message=str(exc) or "GPT action failed",
            detail=str(exc),
            command=command,
        )
    )


def _drop_none_values(value):
    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
    key: str | None = Query(default=None),
    token: str | None = Query(default=None),
):
    """
    If BACKEND_API_KEY is set in .env, every request must include:
    X-API-Key: <BACKEND_API_KEY>

    Custom GPT Actions should be configured with X-API-Key, but accepting a
    bearer token as a fallback makes production diagnosis less brittle when the
    action auth type is accidentally set to Bearer.
    """
    if settings.backend_api_key:
        candidates = [x_api_key, api_key, key, token]
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
def custom_gpt_action_schema(request: Request):
    schema_path = Path(__file__).resolve().parents[1] / "action_schema.gpt-control.yaml"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Action schema not found")
    content = schema_path.read_text(encoding="utf-8")
    content = _set_action_schema_server_url(
        content,
        settings.action_schema_public_url or _request_public_url(request),
    )
    return PlainTextResponse(
        content=content,
        media_type="text/yaml",
        headers={"Cache-Control": "no-store"},
    )


def _request_public_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    proto = proto.split(",", 1)[0].strip()
    host = host.split(",", 1)[0].strip()
    return f"{proto}://{host}".rstrip("/")


def _set_action_schema_server_url(content: str, public_url: str) -> str:
    return re.sub(
        r"(?m)^  - url: https?://[^\s]+$",
        f"  - url: {public_url.rstrip('/')}",
        content,
        count=1,
    )


def _health_payload():
    return {
        "status": "ok",
        "message": "Backend health check passed",
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
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
    operation_id="controlAutoTradingFromGpt",
    summary="Turn auto-trading on or off from Custom GPT",
)
@app.post(
    "/gpt/auto-trading/control/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def control_auto_trading_endpoint(req: GptAutoTradeControlRequest):
    try:
        return _gpt_json(control_auto_trading_from_gpt(req))
    except Exception as exc:
        return _gpt_exception_response(exc, command=str(req.command))


@app.get(
    "/gpt/auto-trading/status",
    response_model=GptAutoTradeControlResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
    operation_id="getGptAutoTradingStatus",
    summary="Get auto-trading status from Custom GPT",
)
@app.get(
    "/gpt/auto-trading/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/auto-trading/status",
    response_model=GptAutoTradeControlResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/auto-trading/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def gpt_auto_trading_status_endpoint():
    try:
        return _gpt_json(
            control_auto_trading_from_gpt(GptAutoTradeControlRequest(command="status"))
        )
    except Exception as exc:
        return _gpt_exception_response(exc, command="status")


@app.post(
    "/gpt/auto-trading/start-paper",
    response_model=GptAutoTradeControlResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
    operation_id="startGptPaperAutoTrading",
    summary="Start paper auto-trading from Custom GPT",
)
@app.post(
    "/gpt/auto-trading/start-paper/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def gpt_start_paper_auto_trading_endpoint(
    payload: dict[str, Any] | None = Body(default=None),
):
    try:
        data = dict(payload or {})
        data.update({"command": "start", "execution_mode": "paper"})
        req = GptAutoTradeControlRequest(**data)
        return _gpt_json(control_auto_trading_from_gpt(req))
    except Exception as exc:
        return _gpt_exception_response(exc, command="start")


@app.post(
    "/universe/scan",
    dependencies=[Depends(verify_api_key)],
    operation_id="scanTradingUniverse",
    summary="Discover and rank auto-trading candidates",
)
def scan_trading_universe(
    req: AutoTradeStartRequest,
    refresh_samples: bool = False,
    sample_limit: int = 20,
):
    scan_result = scan_universe_for_auto_trade(req)
    if not refresh_samples:
        return scan_result
    initialize_universe_db()
    refresh = refresh_edge_training_samples()
    top10 = refresh_top10_performance_if_due(force=True)
    return {
        "status": "success",
        "scan": scan_result,
        "refresh": refresh,
        "top10_performance_refresh": top10,
        "samples": get_edge_training_sample_summary(limit=sample_limit),
    }


@app.post(
    "/universe/scan-and-refresh-samples",
    dependencies=[Depends(verify_api_key)],
    operation_id="scanUniverseAndRefreshSamples",
    summary="Run universe scan then refresh edge training samples",
)
def scan_universe_and_refresh_samples(
    req: AutoTradeStartRequest,
    sample_limit: int = 20,
):
    return scan_trading_universe(
        req,
        refresh_samples=True,
        sample_limit=sample_limit,
    )


@app.get(
    "/universe/latest",
    dependencies=[Depends(verify_api_key)],
    operation_id="getLatestUniverseScan",
    summary="Get the latest stored universe scanner result",
)
def latest_trading_universe():
    return get_latest_universe_scan()

@app.post(
    "/edge-calibration/reset",
    dependencies=[Depends(verify_api_key)],
    operation_id="resetEdgeCalibrationData",
    summary="Reset only edge calibration samples, coefficients, runs, and top10 performance",
)
def reset_edge_calibration_data(payload: dict[str, Any] = Body(...)):
    confirm = str(payload.get("confirm") or "")
    dry_run = bool(payload.get("dry_run", True))

    if confirm != "RESET_EDGE_CALIBRATION":
        raise HTTPException(
            status_code=400,
            detail="confirm must be exactly RESET_EDGE_CALIBRATION",
        )

    db_path = settings.storage_path(settings.edge_calibration_db_path)

    tables = [
        "edge_training_samples",
        "edge_coefficients",
        "edge_calibration_runs",
        "top_candidate_performance",
        "edge_calibration_meta",
    ]

    if not db_path.exists():
        return {
            "status": "empty",
            "message": "Edge calibration DB does not exist.",
            "db_path": str(db_path),
            "dry_run": dry_run,
            "tables": [],
        }

    result_tables = []

    with sqlite3.connect(db_path) as conn:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        for table in tables:
            if table not in existing_tables:
                result_tables.append(
                    {
                        "table": table,
                        "exists": False,
                        "before_count": None,
                        "deleted": 0,
                    }
                )
                continue

            before_count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            deleted = 0
            if not dry_run:
                conn.execute(f"DELETE FROM {table}")
                deleted = int(before_count or 0)

            result_tables.append(
                {
                    "table": table,
                    "exists": True,
                    "before_count": int(before_count or 0),
                    "deleted": deleted,
                }
            )

        if not dry_run:
            conn.commit()
            conn.execute("VACUUM")

    return {
        "status": "success",
        "message": (
            "Dry run completed. No data was deleted."
            if dry_run
            else "Edge calibration DB reset completed."
        ),
        "db_path": str(db_path),
        "dry_run": dry_run,
        "tables": result_tables,
    }

@app.get(
    "/edge-calibration/status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getEdgeCalibrationStatus",
    summary="Get expected-return and risk calibration status",
)
def edge_calibration_status():
    return get_edge_calibration_status()


@app.get(
    "/edge-calibration/samples",
    dependencies=[Depends(verify_api_key)],
    operation_id="getEdgeTrainingSampleSummary",
    summary="Get edge training sample count and realized bps totals",
)
def edge_training_sample_summary(limit: int = 20):
    return get_edge_training_sample_summary(limit=limit)


@app.post(
    "/edge-calibration/refresh-samples",
    dependencies=[Depends(verify_api_key)],
    operation_id="refreshEdgeTrainingSamples",
    summary="Refresh edge training samples and top-10 performance from stored scanner data",
)
def refresh_edge_training_sample_summary(limit: int = 20):
    from app.trading.edge_calibration import label_policy_summary

    initialize_universe_db()
    refresh = refresh_edge_training_samples()
    top10 = refresh_top10_performance_if_due(force=True)
    return {
        "status": "success",
        "refresh": refresh,
        "top10_performance_refresh": top10,
        "samples": get_edge_training_sample_summary(limit=limit),
        "label_policy": label_policy_summary(),
    }


@app.get(
    "/admin/runtime-status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getAdminRuntimeStatus",
    summary="Get read-only admin dashboard runtime status",
)
def admin_runtime_status(limit: int = 20):
    from app.workers.manager import embedded_worker_status
    from app.trading import auto_trading_store

    generated_at = datetime.now().isoformat(timespec="seconds")
    auto_status = control_auto_trading_from_gpt(
        GptAutoTradeControlRequest(command="status")
    )
    samples = get_edge_training_sample_summary(limit=limit)
    latest_universe = get_latest_universe_scan()
    workers = embedded_worker_status()
    raw_active_sessions = auto_trading_store.list_sessions(status="active", limit=50)
    auto_tuning = latest_auto_tuning_recommendation()
    summary = _admin_runtime_summary(
        generated_at=generated_at,
        auto_status=auto_status,
        samples=samples,
        latest_universe=latest_universe,
        workers=workers,
        raw_active_sessions=raw_active_sessions,
    )
    return {
        "status": "success",
        "generated_at": generated_at,
        "summary": summary,
        "auto_trading": auto_status,
        "latest_universe": latest_universe,
        "samples": samples,
        "workers": workers,
        "auto_tuning": auto_tuning,
    }


@app.get(
    "/admin/auto-tuning/latest",
    dependencies=[Depends(verify_api_key)],
    operation_id="getLatestAutoTuningRecommendation",
    summary="Get the latest saved auto-tuning recommendation report",
)
def admin_auto_tuning_latest():
    return latest_auto_tuning_recommendation()


@app.post(
    "/admin/auto-tuning/refresh",
    dependencies=[Depends(verify_api_key)],
    operation_id="refreshAutoTuningRecommendations",
    summary="Generate advisory auto-tuning recommendations from recent outcome attribution",
)
def admin_auto_tuning_refresh():
    return generate_auto_tuning_recommendations(persist=True)


def _admin_runtime_summary(
    *,
    generated_at: str,
    auto_status: dict[str, Any],
    samples: dict[str, Any],
    latest_universe: dict[str, Any],
    workers: dict[str, Any],
    raw_active_sessions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    active_sessions = auto_status.get("active_sessions") or []
    raw_active_sessions = raw_active_sessions or []
    worker_rows = workers.get("workers") or []
    worker_count = len(worker_rows)
    alive_workers = [row for row in worker_rows if row.get("alive")]
    trading_worker_alive = any(
        row.get("name") == "trading_worker" and row.get("alive")
        for row in worker_rows
    )
    diagnostics = samples.get("diagnostics") or {}
    label_policy = samples.get("label_policy") or {}
    top10 = samples.get("top10_performance") or {}
    scan_count = _to_int(diagnostics.get("universe_scan_count"))
    sample_count = _to_int(samples.get("sample_count"))
    candidate_count = _to_int(diagnostics.get("scanner_candidate_history_count"))
    snapshot_count = _to_int(diagnostics.get("universe_price_snapshot_count"))
    latest_scan_time = (
        diagnostics.get("latest_scan_time")
        or latest_universe.get("created_at")
        or latest_universe.get("scan_time")
    )
    latest_next_run_at = active_sessions[0].get("next_run_at") if active_sessions else None
    now_dt = _parse_iso(generated_at) or datetime.now()
    latest_scan_age_seconds = _age_seconds(latest_scan_time, now_dt)
    next_run_lag_seconds = _age_seconds(latest_next_run_at, now_dt)
    stale_after_seconds = _scanner_stale_after_seconds(active_sessions)
    locked_sessions = [
        row
        for row in raw_active_sessions
        if _is_future_time(row.get("locked_until"), now_dt)
    ]
    latest_locked_until = max(
        (str(row.get("locked_until")) for row in locked_sessions if row.get("locked_until")),
        default=None,
    )

    if active_sessions and trading_worker_alive:
        if (
            scan_count > 0
            and latest_scan_age_seconds is not None
            and latest_scan_age_seconds > stale_after_seconds
            and next_run_lag_seconds is not None
            and next_run_lag_seconds > stale_after_seconds
        ):
            scanner_state = "stale"
        elif locked_sessions:
            scanner_state = "running_locked"
        else:
            scanner_state = "running"
    elif active_sessions:
        scanner_state = "worker_down"
    elif scan_count > 0:
        scanner_state = "idle"
    else:
        scanner_state = "not_started"

    if sample_count > 0:
        sample_state = "ready"
    elif scan_count > 0 and label_policy.get("label_at_horizon_end"):
        sample_state = "waiting_for_horizon"
    elif scan_count > 0:
        sample_state = "waiting_for_future_prices"
    else:
        sample_state = "empty"

    return {
        "scanner_state": scanner_state,
        "sample_state": sample_state,
        "active_session_count": len(active_sessions),
        "worker_count": worker_count,
        "alive_worker_count": len(alive_workers),
        "trading_worker_alive": trading_worker_alive,
        "cycle_count": sum(_to_int(row.get("cycle_count")) for row in active_sessions),
        "latest_session_updated_at": active_sessions[0].get("updated_at")
        if active_sessions
        else None,
        "latest_session_next_run_at": latest_next_run_at,
        "next_run_lag_seconds": next_run_lag_seconds,
        "locked_session_count": len(locked_sessions),
        "latest_locked_until": latest_locked_until,
        "latest_scan_id": latest_universe.get("scan_id"),
        "latest_scan_status": latest_universe.get("status"),
        "latest_scan_time": latest_scan_time,
        "latest_scan_age_seconds": latest_scan_age_seconds,
        "scanner_stale_after_seconds": stale_after_seconds,
        "latest_scan_final_count": _to_int(latest_universe.get("final_count")),
        "latest_scan_executable_count": _to_int(
            latest_universe.get("executable_count")
        ),
        "universe_scan_count": scan_count,
        "scanner_candidate_history_count": candidate_count,
        "universe_price_snapshot_count": snapshot_count,
        "sample_count": sample_count,
        "top10_sample_count": _to_int(top10.get("sample_count")),
        "last_training_sample_at": diagnostics.get("last_training_sample_at"),
        "label_horizon_seconds": _to_int(label_policy.get("horizon_seconds")),
        "label_min_age_seconds": _to_int(label_policy.get("min_label_age_seconds")),
    }


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _age_seconds(value: Any, now_dt: datetime) -> int | None:
    parsed = _parse_iso(value)
    if not parsed:
        return None
    return max(0, int((now_dt - parsed).total_seconds()))


def _is_future_time(value: Any, now_dt: datetime) -> bool:
    parsed = _parse_iso(value)
    return bool(parsed and parsed > now_dt)


def _scanner_stale_after_seconds(active_sessions: list[dict[str, Any]]) -> int:
    session_interval = max(
        [_to_int(row.get("interval_seconds")) for row in active_sessions] or [60]
    )
    source_count = max(1, int(settings.universe_scanner_max_source_symbols or 1))
    symbol_interval = max(
        0.0,
        float(settings.universe_scanner_symbol_interval_seconds or 0.0),
    )
    cap = max(0.0, float(settings.universe_scanner_symbol_interval_cap_seconds or 0.0))
    if cap > 0:
        symbol_interval = min(symbol_interval, cap)
    estimated_scan_seconds = int(source_count * symbol_interval) + 300
    return max(900, session_interval * 5, estimated_scan_seconds * 2)


@app.post(
    "/admin/reset-data",
    dependencies=[Depends(verify_api_key)],
    operation_id="resetGeneratedTradingData",
    summary="Delete generated SQLite/cache data from the current backend storage",
)
def reset_generated_trading_data(payload: dict[str, Any] = Body(...)):
    confirm = str(payload.get("confirm") or "")
    if confirm != RESET_CONFIRMATION:
        raise HTTPException(
            status_code=400,
            detail=f"confirm must be exactly {RESET_CONFIRMATION}",
        )
    return reset_trading_data(
        confirm=confirm,
        include_all_data_files=bool(payload.get("include_all_data_files", True)),
        dry_run=bool(payload.get("dry_run", False)),
    )

@app.post(
    "/edge-calibration/reset",
    dependencies=[Depends(verify_api_key)],
    operation_id="resetEdgeCalibrationData",
    summary="Reset only edge calibration samples, coefficients, runs, and top10 performance",
)
def reset_edge_calibration_data(payload: dict[str, Any] = Body(...)):
    confirm = str(payload.get("confirm") or "")
    dry_run = bool(payload.get("dry_run", True))

    if confirm != "RESET_EDGE_CALIBRATION":
        raise HTTPException(
            status_code=400,
            detail="confirm must be exactly RESET_EDGE_CALIBRATION",
        )

    db_path = settings.storage_path(settings.edge_calibration_db_path)

    tables = [
        "edge_training_samples",
        "edge_coefficients",
        "edge_calibration_runs",
        "top_candidate_performance",
        "edge_calibration_meta",
    ]

    if not db_path.exists():
        return {
            "status": "empty",
            "message": "Edge calibration DB does not exist.",
            "db_path": str(db_path),
            "dry_run": dry_run,
            "tables": [],
        }

    result_tables = []

    with sqlite3.connect(db_path) as conn:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        for table in tables:
            if table not in existing_tables:
                result_tables.append(
                    {
                        "table": table,
                        "exists": False,
                        "before_count": None,
                        "deleted": 0,
                    }
                )
                continue

            before_count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            deleted = 0
            if not dry_run:
                conn.execute(f"DELETE FROM {table}")
                deleted = int(before_count or 0)

            result_tables.append(
                {
                    "table": table,
                    "exists": True,
                    "before_count": int(before_count or 0),
                    "deleted": deleted,
                }
            )

        if not dry_run:
            conn.commit()
            conn.execute("VACUUM")

    return {
        "status": "success",
        "message": (
            "Dry run completed. No data was deleted."
            if dry_run
            else "Edge calibration DB reset completed."
        ),
        "db_path": str(db_path),
        "dry_run": dry_run,
        "tables": result_tables,
    }

@app.get(
    "/edge-calibration/gate",
    dependencies=[Depends(verify_api_key)],
    operation_id="getEdgeCalibrationGate",
    summary="Get the calibrated entry-performance gate",
)
def edge_calibration_gate():
    return edge_entry_gate()


@app.post(
    "/edge-calibration/run",
    dependencies=[Depends(verify_api_key)],
    operation_id="runEdgeCalibration",
    summary="Calibrate expected return and risk coefficients from scanner history",
)
def run_edge_calibration():
    return calibrate_edge_model()


@app.get(
    "/auto-trading/sessions",
    response_model=AutoTradeSessionsResponse,
    dependencies=[Depends(verify_api_key)],
    operation_id="listAutoTradingSessions",
    summary="List persistent auto-trading sessions",
)
@app.get(
    "/auto-trading/sessions/",
    response_model=AutoTradeSessionsResponse,
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def list_auto_trading_sessions_endpoint(
    status: str | None = None,
    limit: int = 50,
):
    return list_auto_trading_sessions(status=status, limit=limit)


@app.get(
    "/auto-trading/status",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.get(
    "/auto-trading/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def get_auto_trading_status_compat():
    return control_auto_trading_from_gpt(GptAutoTradeControlRequest(command="status"))


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
    "/auto-trading/stop",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def stop_all_auto_trading_sessions_compat():
    active = list_auto_trading_sessions(status="active", limit=500)["sessions"]
    stopped = [stop_auto_trading(session["session_id"]) for session in active]
    return {
        "status": "success",
        "command": "stop",
        "message": f"Stopped {len(stopped)} active auto-trading session(s)",
        "active_sessions": [],
        "recent_sessions": list_auto_trading_sessions(limit=10)["sessions"],
        "stopped_sessions": stopped,
    }


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


@app.post(
    "/gpt/broker/kis/paper-preflight",
    dependencies=[Depends(verify_api_key)],
    operation_id="preflightGptKisPaperE2E",
    summary="Validate KIS paper connectivity from Custom GPT",
)
@app.post(
    "/gpt/broker/kis/paper-preflight/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.get(
    "/gpt/broker/kis/paper-preflight",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.get(
    "/gpt/broker/kis/paper-preflight/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def gpt_preflight_kis_paper_state(symbol: str = "005930"):
    try:
        result = _drop_none_values(preflight_kis_paper_state(symbol=symbol))
        result.setdefault("status", "success")
        result.setdefault("message", "KIS paper connectivity preflight completed")
        return _gpt_json(result)
    except Exception as exc:
        return _gpt_exception_response(exc)


@app.get(
    "/gpt/broker/kis/account-probe",
    dependencies=[Depends(verify_api_key)],
    operation_id="probeGptKisAccount",
    summary="Probe KIS paper account access without placing an order",
)
@app.get(
    "/gpt/broker/kis/account-probe/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def gpt_probe_kis_account_get(
    symbol: str = "005930",
    product_codes: str = "01,00,02",
    force_token_refresh: bool = False,
):
    try:
        return _gpt_json(
            _probe_kis_account(
                symbol=symbol,
                product_codes=_split_product_codes(product_codes),
                force_token_refresh=force_token_refresh,
            )
        )
    except Exception as exc:
        return _gpt_exception_response(exc)


@app.post(
    "/gpt/broker/kis/account-probe",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/broker/kis/account-probe/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def gpt_probe_kis_account_post(payload: dict[str, Any] | None = Body(default=None)):
    try:
        data = payload or {}
        return _gpt_json(
            _probe_kis_account(
                symbol=str(data.get("symbol") or "005930"),
                product_codes=_split_product_codes(data.get("product_codes") or "01,00,02"),
                force_token_refresh=bool(data.get("force_token_refresh", False)),
            )
        )
    except Exception as exc:
        return _gpt_exception_response(exc)


def _probe_kis_account(
    *,
    symbol: str,
    product_codes: list[str],
    force_token_refresh: bool,
) -> dict[str, Any]:
    client = KisClient(is_paper=True)
    diagnostics = client.runtime_diagnostics()
    token_refresh: dict[str, Any] = {"requested": force_token_refresh}
    if force_token_refresh:
        try:
            token_refresh["issued"] = bool(client.issue_access_token(force_refresh=True))
        except Exception as exc:
            token_refresh.update(
                {
                    "issued": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    quote_probe = _kis_probe_call(lambda: client.get_current_price(symbol))
    balance_probes = [
        {
            "account_product_code": product_code,
            **_kis_probe_call(
                lambda product_code=product_code: client.get_balance(
                    account_product_code=product_code
                )
            ),
        }
        for product_code in product_codes
    ]
    ok_products = [
        row["account_product_code"]
        for row in balance_probes
        if row.get("status") == "ok"
    ]
    status = "success" if ok_products else "error"
    message = (
        f"KIS accepted balance lookup for product code(s): {', '.join(ok_products)}"
        if ok_products
        else "KIS rejected balance lookup for every probed product code"
    )
    return _gpt_success_payload(
        message,
        {
            "status": status,
            "symbol": symbol,
            "diagnostics": diagnostics,
            "token_refresh": token_refresh,
            "quote_probe": quote_probe,
            "balance_probes": balance_probes,
            "accepted_product_codes": ok_products,
        },
    )


def _kis_probe_call(call):
    try:
        data = call()
        return {
            "status": "ok",
            "keys": sorted(data.keys())[:12] if isinstance(data, dict) else [],
        }
    except KisApiError as exc:
        return {
            "status": "error",
            "error_type": "KisApiError",
            "http_status": exc.status_code,
            "error_code": exc.error_code,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _split_product_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value or "").split(",")
    product_codes = []
    for item in raw_values:
        code = "".join(ch for ch in str(item) if ch.isdigit())
        if code and code not in product_codes:
            product_codes.append(code)
    return product_codes or ["01"]


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
    "/gpt/broker/kis/config-status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getGptKisConfigStatus",
    summary="Inspect non-secret KIS runtime configuration from Custom GPT",
)
@app.get(
    "/gpt/broker/kis/config-status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/broker/kis/config-status",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/broker/kis/config-status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def get_gpt_kis_config_status():
    try:
        return _gpt_json(
            _gpt_success_payload(
                "KIS runtime configuration inspected",
                get_kis_config_status(),
            )
        )
    except Exception as exc:
        return _gpt_exception_response(exc)


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
    operation_id="getEmbeddedWorkerStatus",
    summary="Get embedded worker status",
)
@app.get("/worker/status", include_in_schema=False)
def get_embedded_worker_status():
    from app.workers.manager import embedded_worker_status

    return embedded_worker_status()


@app.get(
    "/gpt/workers/status",
    dependencies=[Depends(verify_api_key)],
    operation_id="getGptWorkerStatus",
    summary="Get embedded worker status from Custom GPT",
)
@app.get(
    "/gpt/workers/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.get(
    "/gpt/worker/status",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.get(
    "/gpt/worker/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/workers/status",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/workers/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/worker/status",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
@app.post(
    "/gpt/worker/status/",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def get_gpt_embedded_worker_status():
    try:
        from app.workers.manager import embedded_worker_status

        status = embedded_worker_status()
        return _gpt_json(
            _gpt_success_payload(
                f"{status.get('count', 0)} embedded worker(s) reported",
                status,
            )
        )
    except Exception as exc:
        return _gpt_exception_response(exc)


@app.options("/gpt/{gpt_path:path}", include_in_schema=False)
def gpt_options_fallback(gpt_path: str):
    return Response(
        status_code=204,
        headers={
            "Allow": "GET, POST, OPTIONS",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.options("/gpt", include_in_schema=False)
def gpt_root_options_fallback():
    return gpt_options_fallback("")


@app.api_route(
    "/gpt",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
def gpt_root_action_fallback():
    return gpt_unknown_action_fallback("")


@app.api_route(
    "/gpt/{gpt_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
def gpt_unknown_action_fallback(gpt_path: str):
    path = f"/gpt/{gpt_path}" if gpt_path else "/gpt"
    return _gpt_json(
        _gpt_error_payload(
            error_type="not_found",
            http_status=404,
            message=f"Unknown GPT action path: {path}",
            detail={
                "path": path,
                "known_paths": [
                    "/gpt/auto-trading/control",
                    "/gpt/auto-trading/status",
                    "/gpt/auto-trading/start-paper",
                    "/gpt/workers/status",
                    "/gpt/broker/kis/config-status",
                    "/gpt/broker/kis/paper-preflight",
                    "/gpt/broker/kis/account-probe",
                    "/gpt/health",
                ],
            },
        )
    )


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
    "/admin.html",
    response_class=HTMLResponse,
    include_in_schema=False,
)
@app.get(
    "/admin",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def admin_dashboard():
    path = Path(__file__).resolve().parents[1] / "admin.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="admin.html not found")
    return FileResponse(path, media_type="text/html")


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
