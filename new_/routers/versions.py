"""Versioning Router — Snapshot management, restore, and diff comparison."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
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


def _get_or_create_version_manager(session: dict, session_id: str) -> VersionManager:
    if "version_manager" not in session or not isinstance(session["version_manager"], VersionManager):
        vm = VersionManager(session_id)
        # Seed initial original snapshot
        if "current_df" in session:
            vm.save_version(
                session["current_df"],
                operation_name="Initial Upload",
                description="Dataset uploaded and registered",
                rows_affected=len(session["current_df"])
            )
        session["version_manager"] = vm
    return session["version_manager"]


@router.get("/versions/{session_id}")
async def list_versions(session_id: str):
    """List metadata for all saved version snapshots of dataset."""
    try:
        session = _get_session(session_id)
        vm = _get_or_create_version_manager(session, session_id)
        return JSONResponse(content={"versions": vm.get_history()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")


@router.get("/versions/{session_id}/{version_id}")
async def get_version_details(session_id: str, version_id: str):
    """Get metadata for a specific snapshot version."""
    try:
        session = _get_session(session_id)
        vm = _get_or_create_version_manager(session, session_id)
        meta = vm.get_version_meta(version_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")
        return JSONResponse(content=meta)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch version: {str(e)}")


@router.post("/versions/{session_id}/restore/{version_id}")
async def restore_version(session_id: str, version_id: str):
    """Restore dataset state to a specific historical snapshot version."""
    try:
        session = _get_session(session_id)
        vm = _get_or_create_version_manager(session, session_id)
        df_restored = vm.restore_version(version_id)
        if df_restored is None:
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")

        # Update active dataframe
        session["history"].append(session["current_df"].copy())
        session["current_df"] = df_restored.copy()
        session["operations"].append({
            "timestamp": pd.Timestamp.now().isoformat() if "pd" in globals() else "",
            "operation": f"Restored dataset to version {version_id[:8]}",
            "rows_affected": len(df_restored)
        })

        return JSONResponse(content={
            "success": True,
            "message": f"Successfully restored to version {version_id}",
            "rows": len(df_restored),
            "columns": len(df_restored.columns)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@router.get("/versions/{session_id}/compare")
async def compare_versions_endpoint(session_id: str, v1: str = Query(..., alias="v1"), v2: str = Query(..., alias="v2")):
    """Compare two version snapshots and generate diff report."""
    try:
        session = _get_session(session_id)
        vm = _get_or_create_version_manager(session, session_id)
        diff = vm.compare_versions(v1, v2)
        return JSONResponse(content=diff)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Version comparison failed: {str(e)}")
