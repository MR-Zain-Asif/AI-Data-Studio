"""Data Profiler Router — Comprehensive JSON Profile & ReportLab PDF generation."""

import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from services.profiler import generate_full_profile, generate_pdf_report, get_column_profile

router = APIRouter()

_session_store: dict = None


def set_session_store(store: dict):
    global _session_store
    _session_store = store


def get_session_store() -> dict:
    global _session_store
    if _session_store is None:
        _session_store = {}
    return _session_store


def _get_session(session_id: str) -> dict:
    store = get_session_store()
    if session_id not in store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return store[session_id]


@router.get("/profile/{session_id}")
async def get_profile(session_id: str):
    """Return comprehensive 6-section profiling report in JSON format."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        profile = generate_full_profile(df, session_info=session)
        return JSONResponse(content=profile)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data profiling failed: {str(e)}")


@router.get("/profile/{session_id}/pdf")
async def get_profile_pdf(session_id: str):
    """Download executive multi-page PDF profiling report with charts and cover."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        pdf_bytes = generate_pdf_report(df, session_info=session)
        clean_name = session.get("file_name", "dataset.csv").rsplit(".", 1)[0]
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=profile_{clean_name}.pdf"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF profile generation failed: {str(e)}")


@router.get("/profile/{session_id}/column/{col_name}")
async def get_single_column_profile(session_id: str, col_name: str):
    """Return statistical distribution and anomaly profile for an individual column."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        col_prof = get_column_profile(df, col_name)
        return JSONResponse(content=col_prof)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column profile failed: {str(e)}")
