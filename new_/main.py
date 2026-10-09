"""AI Data Cleaning Studio — Main FastAPI application."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ─────────────────────────
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.schemas import HealthResponse
from routers import (
    upload, analyze, clean, export, ml_model,
    profiler, quality, versions, anomalies, transform, recommendations
)
from routers.agent_router import router as agent_router

# ── Session storage (shared in-memory dict) ──────────────────────
session_store: dict = {}

# ── Inject session store into all routers ────────────────────────
upload.set_session_store(session_store)
analyze.set_session_store(session_store)
clean.set_session_store(session_store)
export.set_session_store(session_store)
ml_model.set_session_store(session_store)
profiler.set_session_store(session_store)
quality.set_session_store(session_store)
versions.set_session_store(session_store)
anomalies.set_session_store(session_store)
transform.set_session_store(session_store)
recommendations.set_session_store(session_store)

# ── Inject shared session store into agent session manager ────────
from services.session_manager import set_agent_session_store
set_agent_session_store(session_store)

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="AI Data Cleaning Studio",
    description="Full-stack intelligent data cleaning, analysis, ML, and autonomous AI agent.",
    version="3.0.0",
)

# ── CORS middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include existing routers ─────────────────────────────────────
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(clean.router, prefix="/api", tags=["Cleaning"])
app.include_router(export.router, prefix="/api", tags=["Export"])
app.include_router(ml_model.router, prefix="/api", tags=["ML Model"])
app.include_router(profiler.router, prefix="/api", tags=["Profiling"])
app.include_router(quality.router, prefix="/api", tags=["Quality"])
app.include_router(versions.router, prefix="/api", tags=["Versions"])
app.include_router(anomalies.router, prefix="/api", tags=["Anomalies"])
app.include_router(transform.router, prefix="/api", tags=["Transform"])
app.include_router(recommendations.router, prefix="/api", tags=["Recommendations"])

# ── Include agent router (WebSocket + REST) ───────────────────────
app.include_router(agent_router)


# ── Health check endpoints ───────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


# ── Active sessions info ─────────────────────────────────────────
@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for sid, data in session_store.items():
        df = data.get("current_df")
        if df is not None:
            sessions.append({
                "session_id": sid,
                "file_name": data.get("file_name", "unknown"),
                "rows": len(df),
                "columns": len(df.columns),
            })
    return {"sessions": sessions}


# ── Serve frontend ───────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse(
        status_code=404,
        content={"detail": "Frontend not found. Place index.html in frontend/ directory."},
    )


# ── Run with uvicorn (Render & local friendly) ───────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
