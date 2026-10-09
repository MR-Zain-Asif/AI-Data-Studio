"""Quality Monitor Router — Real-time 6-dimension metrics, score, and ranked improvements."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from services.monitor import calculate_quality_metrics

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


@router.get("/quality/{session_id}/full")
async def get_full_quality_report(session_id: str):
    """Calculate and return full 6-dimensional data quality audit."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        metrics = calculate_quality_metrics(df)
        return JSONResponse(content=metrics)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality audit failed: {str(e)}")


@router.get("/quality/{session_id}/score")
async def get_quality_score_only(session_id: str):
    """Return lightweight quality score (0-100) and letter grade (A-F)."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        metrics = calculate_quality_metrics(df)
        return JSONResponse(content={
            "overall_score": metrics["overall_score"],
            "grade": metrics["grade"]
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Score calculation failed: {str(e)}")


@router.get("/quality/{session_id}/improvements")
async def get_quality_improvements(session_id: str):
    """Return prioritized, actionable recommendations to improve dataset quality score."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        metrics = calculate_quality_metrics(df)
        return JSONResponse(content={"improvements": metrics["improvements"]})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Improvements fetch failed: {str(e)}")
