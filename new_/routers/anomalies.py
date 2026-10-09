"""Anomaly Detection Router — Statistical, pattern, text, date, and business rule anomalies."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from services.anomaly_service import detect_all_anomalies, fix_all_anomalies
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


@router.get("/anomalies/{session_id}")
async def get_all_anomalies(session_id: str):
    """Scan and return full audit of statistical, pattern, text, date, and business rule anomalies."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        anomalies = detect_all_anomalies(df)
        return JSONResponse(content={
            "total_anomalies": len(anomalies),
            "anomalies": anomalies
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.get("/anomalies/{session_id}/column/{col}")
async def get_column_anomalies(session_id: str, col: str):
    """Scan and return anomalies detected for a specific column."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not in dataset")

        all_anoms = detect_all_anomalies(df)
        col_anoms = [a for a in all_anoms if a.get("column") == col or col in str(a.get("column"))]
        return JSONResponse(content={
            "column": col,
            "anomalies_count": len(col_anoms),
            "anomalies": col_anoms
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column anomaly check failed: {str(e)}")


@router.post("/anomalies/{session_id}/fix-all")
async def fix_anomalies_endpoint(session_id: str):
    """Automatically correct all detected anomalies, save state snapshot, and return remediation log."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]

        fixed_df, affected_rows, applied_fixes = fix_all_anomalies(df)

        # Save snapshot
        vm = session.setdefault("version_manager", VersionManager(session_id))
        vm.save_version(
            fixed_df,
            operation_name="Auto-Fix Anomalies",
            description=f"Remediated anomalies ({len(applied_fixes)} fix types applied)",
            rows_affected=affected_rows
        )
        session["history"].append(session["current_df"].copy())
        session["current_df"] = fixed_df
        session["operations"].append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "operation": f"Fixed anomalies: {'; '.join(applied_fixes[:3])}",
            "rows_affected": affected_rows
        })

        return JSONResponse(content={
            "success": True,
            "rows_affected": affected_rows,
            "fixes_applied": applied_fixes,
            "message": f"Successfully resolved anomalies across {affected_rows} rows"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly auto-fix failed: {str(e)}")
