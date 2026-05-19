from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.app_name,
        "auth_enabled": bool(settings.backend_api_key),
    }


@app.post(
    "/run-full-pipeline",
    response_model=PipelineResponse,
    dependencies=[Depends(verify_api_key)],
)
def run_pipeline(req: PipelineRequest):
    return run_full_pipeline(req)
