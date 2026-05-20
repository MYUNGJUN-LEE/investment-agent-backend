from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from app.services.naver_news import search_naver_news

from app.config import settings
from app.models import PipelineRequest, PipelineResponse
from app.services.pipeline import run_full_pipeline


app = FastAPI(
    title="Investment Agent API",
    version="1.0.0",
    description="Backend API for Custom GPT investment analysis pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: str | None = Header(default=None)):
    """
    If BACKEND_API_KEY is set in .env, every request must include:
    X-API-Key: <BACKEND_API_KEY>
    """
    if settings.backend_api_key:
        if x_api_key != settings.backend_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

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
def health_check():
    return {
        "status": "ok",
        "service": settings.app_name,
        "auth_enabled": bool(settings.backend_api_key),
    }

@app.get("/naver/news")
async def get_naver_news(query: str, display: int = 10):
    return await search_naver_news(query=query, display=display)


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


from fastapi import FastAPI, Header, HTTPException
from app.config import settings
from app.services.pipeline import run_full_pipeline

@app.post("/runMorningBriefing")
async def run_morning_briefing(x_api_key: str | None = Header(default=None)):
    if settings.backend_api_key:
        if x_api_key != settings.backend_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    result = await run_full_pipeline(
        ticker="NVDA",
        market="US",
        strategy="swing",
        recent_hours=72,
        risk_level="medium"
    )

    return {
        "status": "success",
        "message": "Morning briefing completed",
        "result": result
    }