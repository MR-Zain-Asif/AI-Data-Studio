"""Analyze router — statistical profiling, validation rules, PII privacy scanning, dataset comparison, and relationship detection."""

import math
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from models.schemas import ValidationRequest, MaskPIIRequest, CompareRequest
from services.analyzer import DataAnalyzer

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


def _sanitize_val(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _sanitize_dict(d):
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(i) for i in d]
    return _sanitize_val(d)


@router.get("/analyze/{session_id}")
async def analyze_data(session_id: str):
    """Run full data quality analysis on current DataFrame state."""
    try:
        session = _get_session(session_id)
        if "cached_analysis" in session:
            return JSONResponse(content=session["cached_analysis"])

        df = session["current_df"]
        analyzer = DataAnalyzer(
            df,
            dtypes=session.get("dtypes"),
            outliers=session.get("outliers"),
            missing=session.get("missing"),
        )
        report = analyzer.full_report()
        session["dtypes"] = analyzer._cached_dtypes
        session["outliers"] = analyzer._cached_outliers
        session["missing"] = analyzer._cached_missing

        sanitized_report = _sanitize_dict(report)
        # Inject file_name so frontend can display it
        sanitized_report["file_name"] = session.get("file_name", "dataset.csv")
        session["cached_analysis"] = sanitized_report
        return JSONResponse(content=sanitized_report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/stats/{session_id}")
async def get_stats(session_id: str):
    """Get per-column detailed statistics."""
    try:
        session = _get_session(session_id)
        if "cached_stats" in session:
            return JSONResponse(content=session["cached_stats"])

        df = session["current_df"]
        analyzer = DataAnalyzer(
            df,
            dtypes=session.get("dtypes"),
            outliers=session.get("outliers"),
            missing=session.get("missing"),
        )
        stats = analyzer.column_statistics()

        result = {
            "columns": stats,
            "row_count": len(df),
            "column_count": len(df.columns),
        }
        sanitized_result = _sanitize_dict(result)
        session["cached_stats"] = sanitized_result
        return JSONResponse(content=sanitized_result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats calculation failed: {str(e)}")


# ── FEATURE 3: Validation Rules Endpoints ───────────────────────────

COMMON_RULE_TEMPLATES = [
    {
        "template_name": "Age Column Standard",
        "description": "Validates Age is between 0 and 120 and non-empty",
        "rules": [
            {"column": "Age", "rule_type": "min_value", "value": 0, "friendly_name": "Min Age >= 0"},
            {"column": "Age", "rule_type": "max_value", "value": 120, "friendly_name": "Max Age <= 120"},
            {"column": "Age", "rule_type": "not_empty", "value": None, "friendly_name": "Age Not Empty"}
        ]
    },
    {
        "template_name": "Email Address Standard",
        "description": "Validates proper email format and non-empty values",
        "rules": [
            {"column": "Email", "rule_type": "email_format", "value": None, "friendly_name": "Valid Email Format"},
            {"column": "Email", "rule_type": "not_empty", "value": None, "friendly_name": "Email Not Empty"}
        ]
    },
    {
        "template_name": "Phone Number Format",
        "description": "Validates 10-14 digit phone numbers",
        "rules": [
            {"column": "Phone", "rule_type": "phone_format", "value": None, "friendly_name": "Valid Phone Format"}
        ]
    },
    {
        "template_name": "Unique Identifier",
        "description": "Ensures ID column is unique, non-empty, and positive",
        "rules": [
            {"column": "ID", "rule_type": "unique", "value": None, "friendly_name": "Unique ID Values"},
            {"column": "ID", "rule_type": "not_empty", "value": None, "friendly_name": "ID Not Empty"},
            {"column": "ID", "rule_type": "positive", "value": None, "friendly_name": "Positive ID Number"}
        ]
    },
    {
        "template_name": "Financial Price Check",
        "description": "Validates Price is strictly positive and non-empty",
        "rules": [
            {"column": "Price", "rule_type": "positive", "value": None, "friendly_name": "Price > 0"},
            {"column": "Price", "rule_type": "not_empty", "value": None, "friendly_name": "Price Not Empty"}
        ]
    },
    {
        "template_name": "Salary Range Check",
        "description": "Validates Salary is between $1,000 and $10,000,000",
        "rules": [
            {"column": "Salary", "rule_type": "between", "value": [1000, 10000000], "friendly_name": "Salary Range $1k - $10M"}
        ]
    }
]


@router.get("/validate/templates")
async def get_validation_templates():
    """Get common rule templates."""
    return JSONResponse(content={"templates": COMMON_RULE_TEMPLATES})


@router.post("/validate/{session_id}")
async def validate_rules_endpoint(session_id: str, request: ValidationRequest):
    """Run data validation rules against dataset."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        rules_dicts = [rule.model_dump() for rule in request.rules]

        results = DataAnalyzer.validate_rules(df, rules_dicts)

        total_rules = len(results)
        passed_rules = sum(1 for r in results if r["passed"])

        return JSONResponse(content={
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": total_rules - passed_rules,
            "overall_pass": passed_rules == total_rules,
            "results": _sanitize_dict(results)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# ── FEATURE 4: PII Privacy Scanner Endpoints ────────────────────────

@router.get("/privacy/scan/{session_id}")
async def scan_privacy_endpoint(session_id: str):
    """Scan dataset for sensitive PII data."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        scan_res = DataAnalyzer.scan_pii(df)
        return JSONResponse(content=_sanitize_dict(scan_res))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Privacy scan failed: {str(e)}")


@router.post("/privacy/mask/{session_id}")
async def mask_privacy_endpoint(session_id: str, request: MaskPIIRequest):
    """Mask sensitive PII values in a specific column."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        column, pii_type = request.column, request.pii_type

        new_df, masked_count = DataAnalyzer.mask_pii(df, column, pii_type)

        from routers.clean import _save_state, _build_response
        desc = f"Masked {masked_count} PII entries ({pii_type}) in column '{column}'"
        _save_state(session, new_df, desc, masked_count)

        return _build_response(new_df, desc, masked_count)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII masking failed: {str(e)}")


# ── FEATURE 5: Two File Comparison Endpoint ─────────────────────────

@router.post("/compare")
async def compare_datasets_endpoint(request: CompareRequest):
    """Compare two active sessions/datasets."""
    try:
        session1 = _get_session(request.session_id_1)
        session2 = _get_session(request.session_id_2)

        df1 = session1["current_df"]
        df2 = session2["current_df"]

        name1 = request.name1 or session1.get("file_name", "Dataset 1")
        name2 = request.name2 or session2.get("file_name", "Dataset 2")

        comparison = DataAnalyzer.compare_datasets(df1, df2, name1, name2)
        return JSONResponse(content=_sanitize_dict(comparison))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset comparison failed: {str(e)}")


# ── FEATURE 6: Column Relationship Detection Endpoint ───────────────

@router.get("/relationships/{session_id}")
async def get_column_relationships(session_id: str):
    """Analyze correlation matrix and column relationships."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        relationships = DataAnalyzer.analyze_relationships(df)
        return JSONResponse(content=_sanitize_dict(relationships))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relationship analysis failed: {str(e)}")
