"""Data Transformation Router — Pivot, Melt, Transpose, Aggregation, Sampling, Binning, and Math."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import pandas as pd

from services.transform_service import (
    pivot_table, melt_table, transpose, group_and_aggregate,
    smart_sample, bin_numeric_column, apply_math_operation
)
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


def _commit_transform(session: dict, session_id: str, new_df: pd.DataFrame, op_name: str, desc: str):
    vm = session.setdefault("version_manager", VersionManager(session_id))
    vm.save_version(new_df, operation_name=op_name, description=desc, rows_affected=len(new_df))
    session["history"].append(session["current_df"].copy())
    session["current_df"] = new_df
    session["operations"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "operation": f"{op_name}: {desc}",
        "rows_affected": len(new_df)
    })


# ── Request Models ───────────────────────────────────────────────────

class PivotRequest(BaseModel):
    index: Union[str, List[str]]
    columns: Union[str, List[str]]
    values: Union[str, List[str]]
    aggfunc: Optional[str] = "mean"


class MeltRequest(BaseModel):
    id_vars: Optional[List[str]] = None
    value_vars: Optional[List[str]] = None
    var_name: Optional[str] = "variable"
    value_name: Optional[str] = "value"


class AggregateRequest(BaseModel):
    group_by: Union[str, List[str]]
    aggregations: Dict[str, List[str]]


class SampleRequest(BaseModel):
    method: Optional[str] = "random"
    n: Optional[int] = None
    frac: Optional[float] = None
    stratify_col: Optional[str] = None


class BinRequest(BaseModel):
    column: str
    method: Optional[str] = "equal_width"
    bins: Optional[Union[int, List[float]]] = 5
    labels: Optional[List[str]] = None


class MathRequest(BaseModel):
    column: str
    operation: str
    value: Optional[float] = None
    col2: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/transform/{session_id}/pivot")
async def api_pivot(session_id: str, body: PivotRequest):
    """Reshape dataset into a multi-dimensional pivot table."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = pivot_table(df, body.index, body.columns, body.values, body.aggfunc)
        _commit_transform(session, session_id, res_df, "Pivot Table", f"Pivot on {body.columns}")
        return JSONResponse(content={"success": True, "rows": len(res_df), "columns": list(res_df.columns)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pivot failed: {str(e)}")


@router.post("/transform/{session_id}/melt")
async def api_melt(session_id: str, body: MeltRequest):
    """Unpivot wide columns into longitudinal rows."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = melt_table(df, body.id_vars, body.value_vars, body.var_name, body.value_name)
        _commit_transform(session, session_id, res_df, "Melt Table", f"Unpivoted {len(body.value_vars or [])} columns")
        return JSONResponse(content={"success": True, "rows": len(res_df), "columns": list(res_df.columns)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Melt failed: {str(e)}")


@router.post("/transform/{session_id}/transpose")
async def api_transpose(session_id: str):
    """Invert DataFrame axes (rows become columns and columns become rows)."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = transpose(df)
        _commit_transform(session, session_id, res_df, "Transpose", "Inverted rows and columns")
        return JSONResponse(content={"success": True, "rows": len(res_df), "columns": list(res_df.columns)[:20]})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transpose failed: {str(e)}")


@router.post("/transform/{session_id}/aggregate")
async def api_aggregate(session_id: str, body: AggregateRequest):
    """Group by specified keys and compute vector aggregations."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = group_and_aggregate(df, body.group_by, body.aggregations)
        _commit_transform(session, session_id, res_df, "Group & Aggregate", f"Grouped by {body.group_by}")
        return JSONResponse(content={"success": True, "rows": len(res_df), "columns": list(res_df.columns)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")


@router.post("/transform/{session_id}/sample")
async def api_sample(session_id: str, body: SampleRequest):
    """Sample dataset using random, stratified, systematic, head, or tail sampling."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df, info = smart_sample(df, body.method, body.n, body.frac, body.stratify_col)
        _commit_transform(session, session_id, res_df, "Sampling", f"{body.method.title()} sample of {len(res_df)} rows")
        return JSONResponse(content={"success": True, "info": info})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {str(e)}")


@router.post("/transform/{session_id}/bin")
async def api_bin(session_id: str, body: BinRequest):
    """Discretize continuous numeric feature into equal-width, equal-frequency or custom bins."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = bin_numeric_column(df, body.column, body.method, body.bins, body.labels)
        _commit_transform(session, session_id, res_df, "Numeric Binning", f"Binned '{body.column}' ({body.method})")
        return JSONResponse(content={"success": True, "new_column": f"{body.column}_binned", "columns": list(res_df.columns)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binning failed: {str(e)}")


@router.post("/transform/{session_id}/math")
async def api_math(session_id: str, body: MathRequest):
    """Apply vectorized arithmetic, logarithmic, or trigonometric operations."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        res_df = apply_math_operation(df, body.column, body.operation, body.value, body.col2)
        _commit_transform(session, session_id, res_df, "Column Math", f"Applied {body.operation} on '{body.column}'")
        return JSONResponse(content={"success": True, "column": body.column, "message": f"Applied {body.operation} successfully"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Math operation failed: {str(e)}")
