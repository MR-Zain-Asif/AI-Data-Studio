"""Recommendations Router — Prioritized cleaning insights, quick wins, and 1-click critical batch execution."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from services.recommender import generate_smart_recommendations, apply_critical_recommendations
from services.version_service import VersionManager

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


@router.get("/recommendations/{session_id}")
async def get_all_recommendations(session_id: str):
    """Return all rule-based recommendations sorted by priority descending (Critical -> High -> Medium -> Low)."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        task_type = session.get("ml_task_type")
        recs = generate_smart_recommendations(df, task_type=task_type)
        return JSONResponse(content={"recommendations": recs})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation analysis failed: {str(e)}")


@router.get("/recommendations/{session_id}/quick-wins")
async def get_quick_wins(session_id: str):
    """Filter and return only easy, high-impact recommendations (effort <= 2, impact >= 7)."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        recs = generate_smart_recommendations(df)
        quick_wins = [r for r in recs if r.get("effort_score", 10) <= 2 and r.get("impact_score", 0) >= 7]
        return JSONResponse(content={"quick_wins": quick_wins})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick wins fetch failed: {str(e)}")


@router.post("/recommendations/{session_id}/apply-all-critical")
async def apply_critical_endpoint(session_id: str):
    """Execute all Critical recommendations (high missing columns, duplicates, empty columns) in one click."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]

        new_df, affected_rows, applied_logs = apply_critical_recommendations(df)

        if applied_logs:
            vm = session.setdefault("version_manager", VersionManager(session_id))
            vm.save_version(
                new_df,
                operation_name="Apply All Critical",
                description=f"Applied {len(applied_logs)} critical fix(es): {', '.join(applied_logs)}",
                rows_affected=affected_rows
            )
            session["history"].append(session["current_df"].copy())
            session["current_df"] = new_df
            session["operations"].append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "operation": f"Critical fixes: {', '.join(applied_logs)}",
                "rows_affected": affected_rows
            })

        return JSONResponse(content={
            "success": True,
            "applied_fixes": applied_logs,
            "rows_affected": affected_rows,
            "remaining_rows": len(new_df),
            "remaining_cols": len(new_df.columns),
            "message": f"Successfully executed {len(applied_logs)} critical actions"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply critical recommendations: {str(e)}")
