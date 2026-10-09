"""Clean router — cleaning operations, pipeline management, column merge, and local command execution."""

import re
import math
import difflib
import json as _json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import (
    CleanRequest, AssistantRequest, CleanResponse, AssistantResponse,
    PipelineSaveRequest, PipelineRunRequest, MergeColumnsRequest
)
from services.cleaner import DataCleaner, SAVED_PIPELINES, PRESET_PIPELINES
from services.analyzer import DataAnalyzer
from services.groq_client import call_groq

import pandas as pd
import numpy as np

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


def _clear_cache(session: dict):
    session.pop("cached_analysis", None)
    session.pop("cached_stats", None)
    session.pop("dtypes", None)
    session.pop("outliers", None)
    session.pop("missing", None)


def _save_state(session: dict, df, operation_desc: str, rows_affected: int = 0):
    """Save current state to history stack and clear redo stack."""
    session["history"].append(session["current_df"].copy())
    session["redo_stack"] = []
    session["current_df"] = df.copy()
    _clear_cache(session)
    session["operations"].append({
        "timestamp": datetime.now().isoformat(),
        "operation": operation_desc,
        "rows_affected": rows_affected,
    })


def _sanitize_value(v):
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
    return _sanitize_value(d)


def _build_response(df, message: str, rows_affected: int, extra: dict = None):
    try:
        analyzer = DataAnalyzer(df)
        score = analyzer.quality_score()
        if math.isnan(score) or math.isinf(score):
            score = 0.0
    except Exception:
        score = 0.0

    preview_df = df.head(10).copy()
    rows = preview_df.to_dict(orient="records")
    sanitized = []
    for row in rows:
        sanitized.append({k: _sanitize_value(v) for k, v in row.items()})

    result = {
        "success": True,
        "message": message,
        "rows_affected": rows_affected,
        "new_row_count": len(df),
        "new_column_count": len(df.columns),
        "quality_score": score,
        "preview": sanitized,
        "column_names": df.columns.tolist(),
    }

    if extra:
        result.update(_sanitize_dict(extra))

    return JSONResponse(content=result)


@router.post("/clean/{session_id}")
async def clean_data(session_id: str, request: CleanRequest):
    """Apply a cleaning operation to the dataset."""
    try:
        session = _get_session(session_id)
        df = session["current_df"].copy()
        op = request.operation
        col = request.column
        method = request.method or ""
        value = request.value or ""

        if op == "fill_missing":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required for fill_missing")
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            df, n = DataCleaner.fill_missing(df, col, method, value if value else None)
            desc = f"Filled {n} missing values in '{col}' using {method}"

        elif op == "remove_duplicates":
            df, n = DataCleaner.remove_duplicates(df)
            desc = f"Removed {n} duplicate rows"

        elif op == "trim_whitespace":
            df, n = DataCleaner.trim_whitespace(df, col)
            target = f"'{col}'" if col else "all text columns"
            desc = f"Trimmed whitespace in {target} ({n} cells)"

        elif op == "standardize_case":
            case = request.case or method or "lower"
            df, n = DataCleaner.standardize_case(df, col, case)
            target = f"'{col}'" if col else "all text columns"
            desc = f"Standardized case to {case} in {target} ({n} cells)"

        elif op == "remove_empty_columns":
            df, n = DataCleaner.remove_empty_columns(df)
            desc = f"Removed {n} nearly-empty columns"

        elif op == "remove_constant_columns":
            df, n = DataCleaner.remove_constant_columns(df)
            desc = f"Removed {n} constant columns"

        elif op == "fix_outliers":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required")
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            fix_method = method or "cap"
            df, n = DataCleaner.fix_outliers(df, col, fix_method)
            desc = f"Fixed {n} outliers in '{col}' using {fix_method}"

        elif op == "convert_dates":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required")
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            fmt = request.date_format or "%Y-%m-%d"
            df, n = DataCleaner.convert_dates(df, col, fmt)
            desc = f"Converted {n} dates in '{col}' to format {fmt}"

        elif op == "rename_column":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required")
            new_name = request.new_name or value
            if not new_name:
                raise HTTPException(status_code=400, detail="New column name required")
            df, n = DataCleaner.rename_column(df, col, new_name)
            desc = f"Renamed column '{col}' to '{new_name}'"

        elif op == "delete_column":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required")
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            df, n = DataCleaner.delete_column(df, col)
            desc = f"Deleted column '{col}'"

        elif op == "change_dtype":
            if not col:
                raise HTTPException(status_code=400, detail="Column name required")
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            dtype = request.dtype or method
            if not dtype:
                raise HTTPException(status_code=400, detail="Target dtype required")
            df, n = DataCleaner.change_dtype(df, col, dtype)
            desc = f"Converted '{col}' to {dtype}"

        elif op == "fill_all_missing":
            df, n = DataCleaner.fill_all_missing(df)
            desc = f"Smart-filled {n} missing values"

        elif op == "auto_clean":
            df, n, steps = DataCleaner.auto_clean(df)
            step_descriptions = [f"Step {s['step']}: {s['name']} ({s['affected']} affected)" for s in steps if s['affected'] > 0]
            desc = "Auto-clean: " + "; ".join(step_descriptions) if step_descriptions else "Auto-clean: No issues found"
            _save_state(session, df, desc, n)
            return _build_response(df, desc, n, extra={"auto_clean_steps": steps})

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {op}")

        _save_state(session, df, desc, n)
        return _build_response(df, desc, n)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")


# ── FEATURE 1: Pipeline System Endpoints ────────────────────────────

@router.get("/pipeline/list")
async def list_pipelines_endpoint():
    """Get all saved user pipelines and preset pipelines."""
    return JSONResponse(content=DataCleaner.list_pipelines())


@router.get("/pipeline/presets")
async def preset_pipelines_endpoint():
    """Get preset pipeline templates."""
    return JSONResponse(content=PRESET_PIPELINES)


@router.post("/pipeline/save")
async def save_pipeline_endpoint(request: PipelineSaveRequest):
    """Save a user-defined pipeline."""
    try:
        if not request.name or not request.name.strip():
            raise HTTPException(status_code=400, detail="Pipeline name is required")
        if not request.steps:
            # If steps not passed explicitly, attempt to copy operations history from session if session_id provided
            if request.session_id and request.session_id in get_session_store():
                session = get_session_store()[request.session_id]
                ops = session.get("operations", [])
                request.steps = [{"operation": op.get("operation")} for op in ops]

        res = DataCleaner.save_pipeline(request.name.strip(), request.steps)
        return JSONResponse(content={"success": True, "message": f"Pipeline '{request.name}' saved successfully", "pipeline": res})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")


@router.post("/pipeline/run/{session_id}")
async def run_pipeline_endpoint(session_id: str, request: PipelineRunRequest):
    """Run a saved or preset pipeline on the dataset."""
    try:
        session = _get_session(session_id)
        name = request.pipeline_name

        steps = None
        if name in PRESET_PIPELINES:
            steps = PRESET_PIPELINES[name]["steps"]
        elif name in SAVED_PIPELINES:
            steps = SAVED_PIPELINES[name]

        if not steps:
            raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")

        df = session["current_df"].copy()
        total_affected = 0
        total_steps_run = 0

        for step in steps:
            op = step.get("operation")
            col = step.get("column")
            method = step.get("method")
            val = step.get("value")

            n = 0
            if op == "trim_whitespace":
                df, n = DataCleaner.trim_whitespace(df, col)
            elif op == "remove_duplicates":
                df, n = DataCleaner.remove_duplicates(df)
            elif op == "fill_all_missing":
                df, n = DataCleaner.fill_all_missing(df)
            elif op == "fill_missing_mean":
                if col and col in df.columns:
                    df, n = DataCleaner.fill_missing(df, col, "mean")
                else:
                    df, n = DataCleaner.fill_all_missing(df)
            elif op == "fill_missing_mode":
                if col and col in df.columns:
                    df, n = DataCleaner.fill_missing(df, col, "mode")
                else:
                    df, n = DataCleaner.fill_all_missing(df)
            elif op == "fill_missing_median":
                if col and col in df.columns:
                    df, n = DataCleaner.fill_missing(df, col, "median")
                else:
                    df, n = DataCleaner.fill_all_missing(df)
            elif op == "fix_outliers_cap":
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df, cn = DataCleaner.fix_outliers(df, c, "cap")
                        n += cn
            elif op == "fix_outliers_remove":
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df, cn = DataCleaner.fix_outliers(df, c, "remove")
                        n += cn
            elif op in ("fix_outliers_all", "fix_outliers"):
                fix_m = method or "cap"
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df, cn = DataCleaner.fix_outliers(df, c, fix_m)
                        n += cn
            elif op == "standardize_case_title":
                df, n = DataCleaner.standardize_case(df, None, "title")
            elif op in ("standardize_case_all", "standardize_case"):
                case_type = step.get("case") or method or "lower"
                df, n = DataCleaner.standardize_case(df, col, case_type)
            elif op == "remove_constant_columns":
                df, n = DataCleaner.remove_constant_columns(df)
            elif op == "remove_empty_columns":
                df, n = DataCleaner.remove_empty_columns(df)

            total_affected += n
            total_steps_run += 1
            _save_state(session, df, f"Pipeline step {total_steps_run} [{name}]: {op}", n)

        analyzer = DataAnalyzer(df)
        new_score = analyzer.quality_score()

        return JSONResponse(content={
            "success": True,
            "message": f"Successfully ran pipeline '{name}' ({total_steps_run} steps completed)",
            "total_steps_run": total_steps_run,
            "total_rows_affected": total_affected,
            "new_quality_score": new_score,
            "preview": _sanitize_dict(df.head(10).to_dict(orient="records")),
            "column_names": df.columns.tolist()
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


# ── FEATURE 9: Column Merger Endpoints ──────────────────────────────

@router.get("/columns/similar/{session_id}")
async def get_similar_columns(session_id: str):
    """Find similar column pairs for smart merging."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        pairs = DataCleaner.find_similar_columns(df)
        return JSONResponse(content={"similar_pairs": pairs})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar columns: {str(e)}")


@router.post("/columns/merge/{session_id}")
async def merge_columns_endpoint(session_id: str, request: MergeColumnsRequest):
    """Merge two columns using selected strategy."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        col1, col2, strategy, new_name = request.col1, request.col2, request.strategy, request.new_name

        if col1 not in df.columns or col2 not in df.columns:
            raise HTTPException(status_code=400, detail="One or both target columns do not exist")

        df, rows_affected = DataCleaner.merge_columns(df, col1, col2, strategy, new_name)
        desc = f"Merged '{col1}' and '{col2}' into '{new_name}' using '{strategy}' strategy"

        _save_state(session, df, desc, rows_affected)
        return _build_response(df, desc, rows_affected)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column merge failed: {str(e)}")


# ── Undo / Redo ──────────────────────────────────────────────────

@router.post("/undo/{session_id}")
async def undo(session_id: str):
    try:
        session = _get_session(session_id)
        history = session["history"]

        if len(history) < 1:
            raise HTTPException(status_code=400, detail="Nothing to undo")

        session["redo_stack"].append(session["current_df"].copy())
        df = history.pop().copy()
        session["current_df"] = df
        _clear_cache(session)

        if session["operations"]:
            session["operations"].pop()

        return _build_response(df, "Undo successful", 0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Undo failed: {str(e)}")


@router.post("/redo/{session_id}")
async def redo(session_id: str):
    try:
        session = _get_session(session_id)
        redo_stack = session["redo_stack"]

        if not redo_stack:
            raise HTTPException(status_code=400, detail="Nothing to redo")

        session["history"].append(session["current_df"].copy())
        df = redo_stack.pop().copy()
        session["current_df"] = df
        _clear_cache(session)

        return _build_response(df, "Redo successful", 0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redo failed: {str(e)}")


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    try:
        session = _get_session(session_id)
        return {
            "operations": session.get("operations", []),
            "can_undo": len(session["history"]) >= 1,
            "can_redo": len(session["redo_stack"]) > 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History failed: {str(e)}")


# ── FEATURE 10: Natural Language Assistant ──────────────────────────

ALL_COMMAND_CANONICAL = [
    "remove duplicates",
    "fill missing [column]",
    "fill all missing",
    "delete column [name]",
    "rename [old] to [new]",
    "trim whitespace",
    "uppercase [column]",
    "lowercase [column]",
    "title case [column]",
    "fix outliers [column]",
    "remove empty columns",
    "remove constant columns",
    "auto clean",
    "show stats",
    "show quality score",
    "find duplicates",
    "find missing",
    "scan privacy",
    "validate data",
    "export csv",
    "export excel",
    "export json",
    "run pipeline [name]",
    "save pipeline [name]",
]


def _fuzzy_match_command(input_text: str, columns: list[str]) -> tuple[dict | None, int, list[str]]:
    """Match natural language command to action, returning (parsed_dict, confidence, suggestions)."""
    text = input_text.strip().lower()
    text_clean = re.sub(r'[^\w\s]', '', text)

    # 1: Pipeline commands
    m = re.search(r'(?:run|apply)\s+pipeline\s+(.+)', text)
    if m:
        pname = m.group(1).strip()
        return {"action_type": "run_pipeline", "pipeline_name": pname}, 95, []

    m = re.search(r'(?:save|create)\s+pipeline\s+(.+)', text)
    if m:
        pname = m.group(1).strip()
        return {"action_type": "save_pipeline", "pipeline_name": pname}, 95, []

    # 2: Analysis/Export commands
    if any(k in text for k in ["show stats", "show statistics", "column stats"]):
        return {"action_type": "show_stats"}, 95, []
    if any(k in text for k in ["quality score", "what is my score", "data quality"]):
        return {"action_type": "show_quality"}, 95, []
    if any(k in text for k in ["find duplicates", "show duplicates", "how many duplicates"]):
        return {"action_type": "show_duplicates"}, 95, []
    if any(k in text for k in ["find missing", "show missing", "how many missing"]):
        return {"action_type": "show_missing"}, 95, []
    if any(k in text for k in ["scan privacy", "check pii", "find personal data", "privacy scan"]):
        return {"action_type": "scan_privacy"}, 95, []
    if any(k in text for k in ["validate data", "check rules", "run validation"]):
        return {"action_type": "validate_data"}, 95, []
    if any(k in text for k in ["export csv", "download csv", "save as csv"]):
        return {"action_type": "export_csv"}, 95, []
    if any(k in text for k in ["export excel", "download excel", "save as xlsx"]):
        return {"action_type": "export_excel"}, 95, []
    if any(k in text for k in ["export json", "download json"]):
        return {"action_type": "export_json"}, 95, []

    # 3: Clean commands without column
    if any(k in text for k in ["remove duplicates", "delete duplicates", "drop duplicate rows", "remove duplicate rows", "remove duplicate", "drop duplicates", "delete duplicate rows"]):
        return {"action_type": "clean", "operation": "remove_duplicates"}, 95, []
    if any(k in text for k in ["remove empty columns", "delete empty columns", "drop empty columns"]):
        return {"action_type": "clean", "operation": "remove_empty_columns"}, 95, []
    if any(k in text for k in ["remove constant columns", "delete useless columns"]):
        return {"action_type": "clean", "operation": "remove_constant_columns"}, 95, []
    if any(k in text for k in ["auto clean", "clean everything", "fix everything", "clean all"]):
        return {"action_type": "clean", "operation": "auto_clean"}, 95, []
    if any(k in text for k in ["fill all missing", "fix all empty", "fill everything", "fill all"]):
        return {"action_type": "clean", "operation": "fill_all_missing"}, 95, []
    if any(k in text for k in ["trim whitespace", "remove spaces", "clean spaces"]):
        return {"action_type": "clean", "operation": "trim_whitespace"}, 95, []

    # Helper: Find closest column name in text
    def find_col_in_text(t):
        for col in columns:
            if col.lower() in t:
                return col
        # difflib fallback
        words = t.split()
        for w in words:
            matches = difflib.get_close_matches(w, [c.lower() for c in columns], n=1, cutoff=0.6)
            if matches:
                for c in columns:
                    if c.lower() == matches[0]:
                        return c
        return None

    # 4: Column specific cleaning operations
    # Fill missing
    if "fill" in text or "fix missing" in text or "fill empty" in text:
        col = find_col_in_text(text)
        if col:
            return {"action_type": "clean", "operation": "fill_missing", "column": col, "method": "mean"}, 90, []

    # Rename
    m = re.search(r'rename\s+(.+?)\s+to\s+(.+)', text) or re.search(r'change\s+(.+?)\s+name\s+to\s+(.+)', text)
    if m:
        old_col = find_col_in_text(m.group(1)) or m.group(1).strip()
        new_col = m.group(2).strip()
        return {"action_type": "clean", "operation": "rename_column", "column": old_col, "value": new_col, "new_name": new_col}, 90, []

    # Delete column
    if "delete" in text or "remove" in text or "drop" in text:
        col = find_col_in_text(text)
        if col:
            return {"action_type": "clean", "operation": "delete_column", "column": col}, 90, []

    # Case standardization
    if "upper" in text or "uppercase" in text or "capitalize" in text:
        col = find_col_in_text(text)
        return {"action_type": "clean", "operation": "standardize_case", "column": col, "method": "upper"}, 90, []
    if "lower" in text or "lowercase" in text:
        col = find_col_in_text(text)
        return {"action_type": "clean", "operation": "standardize_case", "column": col, "method": "lower"}, 90, []
    if "title" in text or "proper case" in text:
        col = find_col_in_text(text)
        return {"action_type": "clean", "operation": "standardize_case", "column": col, "method": "title"}, 90, []

    # Fix outliers
    if "outlier" in text:
        col = find_col_in_text(text)
        if col:
            return {"action_type": "clean", "operation": "fix_outliers", "column": col, "method": "cap"}, 90, []

    # 5: Low confidence difflib match for suggestions
    closest = difflib.get_close_matches(text_clean, ALL_COMMAND_CANONICAL, n=3, cutoff=0.2)
    if not closest:
        closest = ALL_COMMAND_CANONICAL[:3]

    if closest and difflib.SequenceMatcher(None, text_clean, closest[0]).ratio() > 0.45:
        return None, 45, closest

    return None, 15, closest


def _get_closest_suggestions(text: str, n: int = 3) -> list[str]:
    clean_t = re.sub(r'[^\w\s]', '', text.strip().lower())
    m = difflib.get_close_matches(clean_t, ALL_COMMAND_CANONICAL, n=n, cutoff=0.2)
    return m if len(m) >= n else (m + ALL_COMMAND_CANONICAL[:n])[:n]


def _parse_nl_command(command: str, columns: list[str]) -> dict | None:
    res, conf, _ = _fuzzy_match_command(command, columns)
    if not res:
        return None
    if res.get("action_type") == "clean":
        return res
    if res.get("action_type") in ("show_stats", "export_csv"):
        return {"operation": res["action_type"]}
    return {"operation": res.get("operation", res.get("action_type"))}


@router.post("/assistant-legacy/{session_id}")
async def assistant_legacy(session_id: str, request: AssistantRequest):
    """Local deterministic command runner."""
    try:
        store = get_session_store()
        if session_id not in store:
            return JSONResponse(content={
                "success": False,
                "interpreted_as": "no_session",
                "confidence": 0,
                "result_message": "Active dataset session not found. Please upload a dataset first.",
                "message": "Active dataset session not found. Please upload a dataset first.",
                "suggestion": "Upload a dataset first."
            })

        session = store[session_id]
        df = session["current_df"]
        columns = df.columns.tolist()

        # Local fuzzy matcher for quick commands
        parsed, confidence, suggestions = _fuzzy_match_command(request.command, columns)

        if not parsed or confidence < 60:
            cmd_lower = request.command.lower()
            if any(k in cmd_lower for k in ["explain", "describe", "what", "how", "tell", "summary", "steps", "history", "help", "detail"]):
                applied_ops = session.get("applied_operations", [])
                history_desc = ", ".join([f"{op.get('operation', 'op')} on {op.get('column', 'dataset')}" for op in applied_ops]) if applied_ops else "No transformations applied yet."
                sys_prompt = "You are DataStudio AI, an expert data scientist assistant. Explain dataset cleaning and analysis steps concisely."
                user_prompt = f"Dataset shape: {df.shape}. Columns: {columns}. History: {history_desc}. User query: {request.command}"
                groq_resp = call_groq(sys_prompt, user_prompt, fast=True)
                if not groq_resp or "not configured" in groq_resp or "unavailable" in groq_resp:
                    groq_resp = f"Dataset cleaning steps so far: {history_desc}. Ready to help you clean, transform, or analyze your dataset."
                return JSONResponse(content={
                    "success": True,
                    "interpreted_as": "Groq LLM Assistant",
                    "confidence": 95,
                    "result_message": groq_resp,
                    "rows_affected": 0,
                    "suggestion": None
                })

            sugg_str = f"Did you mean: '{suggestions[0]}'?" if suggestions else "Please try commands like 'auto clean', 'remove duplicates', or 'show stats'."
            return JSONResponse(content={
                "success": False,
                "interpreted_as": "unknown",
                "confidence": confidence,
                "result_message": f"I couldn't confidently execute '{request.command}'.",
                "suggestion": sugg_str,
                "suggestions": suggestions,
            })

        action_type = parsed["action_type"]

        if action_type == "clean":
            clean_req = CleanRequest(
                operation=parsed["operation"],
                column=parsed.get("column"),
                method=parsed.get("method"),
                value=parsed.get("value"),
                new_name=parsed.get("new_name"),
            )
            res = await clean_data(session_id, clean_req)
            res_data = _json.loads(res.body.decode())
            return JSONResponse(content={
                "success": True,
                "interpreted_as": parsed["operation"],
                "confidence": confidence,
                "result_message": res_data.get("message", "Cleaning operation completed"),
                "rows_affected": res_data.get("rows_affected", 0),
                "operation_result": res_data,
                "suggestion": None
            })

        elif action_type == "run_pipeline":
            run_req = PipelineRunRequest(pipeline_name=parsed["pipeline_name"])
            res = await run_pipeline_endpoint(session_id, run_req)
            res_data = _json.loads(res.body.decode())
            return JSONResponse(content={
                "success": True,
                "interpreted_as": f"run_pipeline({parsed['pipeline_name']})",
                "confidence": confidence,
                "result_message": res_data.get("message", "Pipeline executed"),
                "rows_affected": res_data.get("total_rows_affected", 0),
                "operation_result": res_data,
                "suggestion": None
            })

        elif action_type == "save_pipeline":
            save_req = PipelineSaveRequest(session_id=session_id, name=parsed["pipeline_name"], steps=[])
            res = await save_pipeline_endpoint(save_req)
            res_data = _json.loads(res.body.decode())
            return JSONResponse(content={
                "success": True,
                "interpreted_as": f"save_pipeline({parsed['pipeline_name']})",
                "confidence": confidence,
                "result_message": res_data.get("message", "Pipeline saved"),
                "rows_affected": 0,
                "suggestion": None
            })

        elif action_type == "show_stats":
            analyzer = DataAnalyzer(df)
            stats = analyzer.column_statistics()
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "show_stats",
                "confidence": confidence,
                "result_message": f"Dataset has {len(df)} rows and {len(df.columns)} columns across {len(stats)} analyzed features.",
                "rows_affected": 0,
                "suggestion": None
            })

        elif action_type == "show_quality":
            analyzer = DataAnalyzer(df)
            score = analyzer.quality_score()
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "show_quality",
                "confidence": confidence,
                "result_message": f"Current dataset quality score is {score}/100.",
                "rows_affected": 0,
                "suggestion": None
            })

        elif action_type == "show_duplicates":
            analyzer = DataAnalyzer(df)
            dups = analyzer.duplicate_rows()
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "show_duplicates",
                "confidence": confidence,
                "result_message": f"Found {dups['count']} duplicate rows in current dataset.",
                "rows_affected": dups['count'],
                "suggestion": "Type 'remove duplicates' to clean them." if dups['count'] > 0 else None
            })

        elif action_type == "show_missing":
            analyzer = DataAnalyzer(df)
            missing = analyzer.missing_values()
            total_m = sum(v["count"] for v in missing.values())
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "show_missing",
                "confidence": confidence,
                "result_message": f"Found {total_m} missing cells across {len(missing)} columns.",
                "rows_affected": total_m,
                "suggestion": "Type 'fill all missing' to smart-impute values." if total_m > 0 else None
            })

        elif action_type == "scan_privacy":
            scan_res = DataAnalyzer.scan_pii(df)
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "scan_privacy",
                "confidence": confidence,
                "result_message": f"Privacy Scan found {scan_res['total_pii_found']} PII risks ({scan_res['high_risk_count']} high risk).",
                "rows_affected": scan_res['total_pii_found'],
                "suggestion": None
            })

        elif action_type == "validate_data":
            return JSONResponse(content={
                "success": True,
                "interpreted_as": "validate_data",
                "confidence": confidence,
                "result_message": "Switching to Validation screen to execute rules.",
                "rows_affected": 0,
                "suggestion": None
            })

        elif action_type in ("export_csv", "export_excel", "export_json"):
            fmt = action_type.split("_")[1].upper()
            return JSONResponse(content={
                "success": True,
                "interpreted_as": action_type,
                "confidence": confidence,
                "result_message": f"Ready to export dataset as {fmt}. Use topbar export actions to download.",
                "rows_affected": 0,
                "suggestion": None
            })

        else:
            return JSONResponse(content={
                "success": False,
                "interpreted_as": "unknown",
                "confidence": 0,
                "result_message": f"Unrecognized action type for command: {request.command}",
                "suggestion": None
            })

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "interpreted_as": "error", "confidence": 0, "result_message": f"Error: {e.detail}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "interpreted_as": "error", "confidence": 0, "result_message": f"Assistant error: {str(e)}"})


@router.get("/assistant/{session_id}/suggestions")
async def get_assistant_suggestions(session_id: str):
    store = get_session_store()
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
    session = store[session_id]
    df = session["current_df"]
    
    missing_cnt = int(df.isnull().sum().sum())
    dup_cnt = int(df.duplicated().sum())
    suggs = []
    if dup_cnt > 0:
        suggs.append(f"Remove {dup_cnt} duplicate rows")
    if missing_cnt > 0:
        suggs.append("Fill missing values across dataset")
    suggs.append("Auto clean dataset")
    suggs.append("Scan dataset for PII privacy risks")
    
    return JSONResponse(content={
        "success": True,
        "proactive_suggestion": suggs[0] if suggs else "Dataset looks clean!",
        "suggestions": suggs
    })


@router.post("/assistant/{session_id}/chat")
async def assistant_chat(session_id: str, payload: dict):
    message = payload.get("message", payload.get("command", ""))
    req = AssistantRequest(command=message)
    res = await assistant(session_id, req)
    res_data = _json.loads(res.body.decode())
    
    store = get_session_store()
    if session_id in store:
        hist = store[session_id].setdefault("history", [])
        hist.append({"role": "user", "content": message})
        hist.append({"role": "assistant", "content": res_data.get("result_message", "")})
    
    return JSONResponse(content={
        "success": res_data.get("success", True),
        "message": res_data.get("result_message", ""),
        "reply": res_data.get("result_message", ""),
        "auto_executed": res_data.get("success", False),
        "operation_result": res_data.get("operation_result")
    })


@router.get("/assistant/{session_id}/history")
async def get_assistant_history(session_id: str):
    store = get_session_store()
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
    hist = store[session_id].get("history", [])
    
    clean_hist = []
    for item in hist:
        if isinstance(item, dict):
            clean_item = {}
            for k, v in item.items():
                if isinstance(v, pd.DataFrame):
                    clean_item[k] = f"DataFrame({v.shape[0]}x{v.shape[1]})"
                elif hasattr(v, "to_dict"):
                    clean_item[k] = str(v)
                elif isinstance(v, (str, int, float, bool, type(None))):
                    clean_item[k] = v
                else:
                    clean_item[k] = str(v)
            clean_hist.append(clean_item)
        elif isinstance(item, pd.DataFrame):
            clean_hist.append(f"DataFrame({item.shape[0]}x{item.shape[1]})")
        else:
            clean_hist.append(str(item))

    return JSONResponse(content={
        "success": True,
        "history": clean_hist
    })


@router.post("/assistant/{session_id}/clear")
async def clear_assistant_history(session_id: str):
    store = get_session_store()
    if session_id in store:
        store[session_id]["history"] = []
    return JSONResponse(content={
        "success": True,
        "message": "Assistant chat history cleared."
    })
