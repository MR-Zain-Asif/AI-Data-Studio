"""Autonomous AI Agent Orchestrator for AI Data Studio.
Decomposes complex data goals into multi-step execution plans and runs them with rollback & telemetry.
"""

import os
import re
import json
import math
import httpx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from services.analyzer import DataAnalyzer
from services.cleaner import DataCleaner
from services.monitor import calculate_quality_metrics
from services.version_service import VersionManager
from services.ml_service import train_models, detect_target_column

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"


def _sanitize(val):
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize(v) for v in val]
    return val


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Extract fast dataset metadata for AI prompt injection."""
    cols_meta = []
    for c in df.columns[:25]:
        dtype = str(df[c].dtype)
        null_count = int(df[c].isnull().sum())
        nunique = int(df[c].nunique(dropna=True))
        cols_meta.append({"column": c, "dtype": dtype, "nulls": null_count, "unique": nunique})

    quality = calculate_quality_metrics(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "quality_score": quality.get("overall_score", 0),
        "missing_cells": quality.get("completeness", {}).get("missing_cells", 0),
        "duplicate_rows": quality.get("uniqueness", {}).get("duplicate_rows", 0),
        "outlier_count": quality.get("accuracy", {}).get("outlier_count", 0),
        "columns_list": cols_meta
    }


async def plan_goal_steps(user_goal: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Use Groq LLM to decompose user natural language prompt into atomic data actions."""
    meta = summarize_dataset(df)

    system_prompt = f"""You are an Autonomous Data Scientist Agent.
Given the dataset metadata and the user's objective, construct an optimal sequential JSON plan of operations.

DATASET CONTEXT:
- Dimensions: {meta['rows']} rows × {meta['columns']} columns
- Quality Health: {meta['quality_score']}/100
- Missing values: {meta['missing_cells']} cells
- Duplicates: {meta['duplicate_rows']} rows
- Outliers: {meta['outlier_count']}
- Columns: {json.dumps(meta['columns_list'])}

SUPPORTED STEP ACTIONS & FORMAT:
1. {{"action": "remove_duplicates", "params": {{}}, "description": "Remove all duplicate rows"}}
2. {{"action": "fill_missing", "params": {{"column": "col_name_or_all", "method": "median|mean|mode|knn|constant"}}, "description": "Fill missing values"}}
3. {{"action": "fix_outliers", "params": {{"column": "col_name_or_all", "method": "cap|remove"}}, "description": "Cap extreme numerical outliers"}}
4. {{"action": "trim_whitespace", "params": {{}}, "description": "Trim whitespace in text columns"}}
5. {{"action": "create_column", "params": {{"name": "new_col", "expression": "age > 30"}}, "description": "Create calculated flag/expression"}}
6. {{"action": "drop_column", "params": {{"column": "col_name"}}, "description": "Drop unneeded column"}}
7. {{"action": "train_model", "params": {{"target_column": "target_col"}}, "description": "Train AutoML models on target"}}
8. {{"action": "generate_chart", "params": {{"chart_type": "bar|line|scatter|pie|hist", "x": "col1", "y": "col2"}}, "description": "Generate visual chart"}}
9. {{"action": "summarize_insights", "params": {{}}, "description": "Generate executive data narrative"}}

OUTPUT FORMAT:
Return ONLY a valid JSON array of plan step objects. No markdown formatting, no commentary.
Example:
[
  {{"step": 1, "action": "fill_missing", "params": {{"column": "all", "method": "median"}}, "description": "Impute all missing numerical values"}},
  {{"step": 2, "action": "fix_outliers", "params": {{"column": "all", "method": "cap"}}, "description": "Winsorize numerical outliers"}}
]"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Goal: {user_goal}"}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    raw_content = ""
    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                payload["model"] = FALLBACK_MODEL
                resp = await client.post(GROQ_API_URL, headers=headers, json=payload)

            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Extract JSON Array
    parsed_steps = []
    try:
        match = re.search(r"\[\s*\{.*\}\s*\]", raw_content, re.DOTALL)
        if match:
            parsed_steps = json.loads(match.group(0))
        else:
            parsed_steps = json.loads(raw_content)
    except Exception:
        # Heuristic fallback if LLM array parse fails
        goal_lower = user_goal.lower()
        step_id = 1
        if "duplicate" in goal_lower:
            parsed_steps.append({"step": step_id, "action": "remove_duplicates", "params": {}, "description": "Remove duplicate rows"})
            step_id += 1
        if "missing" in goal_lower or "null" in goal_lower or "clean" in goal_lower:
            parsed_steps.append({"step": step_id, "action": "fill_missing", "params": {"column": "all", "method": "median"}, "description": "Fill missing values with smart median"})
            step_id += 1
        if "outlier" in goal_lower:
            parsed_steps.append({"step": step_id, "action": "fix_outliers", "params": {"column": "all", "method": "cap"}, "description": "Cap outliers with IQR bounds"})
            step_id += 1
        if "train" in goal_lower or "predict" in goal_lower or "model" in goal_lower:
            target = detect_target_column(df)
            parsed_steps.append({"step": step_id, "action": "train_model", "params": {"target_column": target}, "description": f"Train AutoML model on {target}"})
            step_id += 1
        if not parsed_steps:
            parsed_steps.append({"step": 1, "action": "summarize_insights", "params": {}, "description": "Analyze dataset and summarize insights"})

    for idx, s in enumerate(parsed_steps, 1):
        s["step"] = idx
        s["status"] = "pending"

    return parsed_steps


def execute_single_step(session: dict, session_id: str, step: dict) -> dict:
    """Execute a single atomic step of the AI agent pipeline with state persistence."""
    df = session["current_df"]
    action = step.get("action")
    params = step.get("params", {})
    affected = 0
    details = ""
    updated_df = df.copy()
    viz_payload = None

    q_before = calculate_quality_metrics(df)["overall_score"]

    if action == "remove_duplicates":
        updated_df = updated_df.drop_duplicates()
        affected = len(df) - len(updated_df)
        details = f"Removed {affected} duplicate rows"

    elif action == "fill_missing":
        col = params.get("column", "all")
        method = params.get("method", "median")
        if col == "all" or col not in updated_df.columns:
            updated_df, affected = DataCleaner.fill_all_missing(updated_df, strategy=method if method in ["mean", "median", "mode", "zero"] else "smart")
            details = f"Imputed {affected} missing cells across all columns"
        else:
            if method == "knn":
                updated_df, affected = DataCleaner.knn_impute(updated_df, col)
            else:
                updated_df, affected = DataCleaner.fill_missing(updated_df, col, method)
            details = f"Filled {affected} missing cells in '{col}' via {method}"

    elif action == "fix_outliers":
        col = params.get("column", "all")
        method = params.get("method", "cap")
        if col == "all" or col not in updated_df.columns:
            updated_df, affected = DataCleaner.fix_outliers_all(updated_df, method=method)
            details = f"{method.title()} outliers in numeric columns ({affected} values adjusted)"
        else:
            updated_df, affected = DataCleaner.fix_outliers(updated_df, col, method=method)
            details = f"{method.title()} outliers in '{col}' ({affected} values adjusted)"

    elif action == "trim_whitespace":
        updated_df, affected = DataCleaner.trim_whitespace(updated_df)
        details = f"Trimmed whitespace in {affected} text columns"

    elif action == "drop_column":
        col = params.get("column")
        if col in updated_df.columns:
            updated_df = updated_df.drop(columns=[col])
            affected = len(updated_df)
            details = f"Dropped column '{col}'"

    elif action == "create_column":
        name = params.get("name", "new_feature")
        expr = params.get("expression", "")
        try:
            # Safe evaluation for basic comparisons / operations
            # e.g., updated_df.eval()
            updated_df[name] = updated_df.eval(expr)
            affected = len(updated_df)
            details = f"Created calculated feature '{name}' = {expr}"
        except Exception as e:
            details = f"Expression evaluation notice: {str(e)}"

    elif action == "train_model":
        target = params.get("target_column") or detect_target_column(updated_df)
        if target and target in updated_df.columns:
            results = train_models(updated_df, target_column=target, session_id=session_id)
            session["ml_results"] = results
            best_m = results.get("best_model_name", "Model")
            best_score = results.get("best_score", 0)
            details = f"AutoML trained successfully! Best: {best_m} (Score: {round(best_score, 3)})"
        else:
            details = "Target column not detected for AutoML."

    elif action == "generate_chart":
        chart_type = params.get("chart_type", "bar")
        x = params.get("x") or (list(updated_df.columns)[0] if len(updated_df.columns) > 0 else "")
        y = params.get("y") or (list(updated_df.select_dtypes(include=[np.number]).columns)[0] if len(updated_df.select_dtypes(include=[np.number]).columns) > 0 else "")
        details = f"Rendered interactive {chart_type} chart ({x} vs {y})"
        viz_payload = {"chart_type": chart_type, "x": x, "y": y}

    elif action == "summarize_insights":
        details = "Synthesized dataset health and key metric distributions."

    # Update session data
    q_after = calculate_quality_metrics(updated_df)["overall_score"]
    vm = session.setdefault("version_manager", VersionManager(session_id))
    vm.save_version(
        updated_df,
        operation_name=f"Copilot: {action}",
        description=step.get("description", details),
        rows_affected=affected
    )
    session["history"].append(session["current_df"].copy())
    session["current_df"] = updated_df
    session["operations"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "operation": f"AI Copilot: {action}",
        "rows_affected": affected
    })

    return _sanitize({
        "success": True,
        "action": action,
        "details": details,
        "rows_affected": affected,
        "quality_before": q_before,
        "quality_after": q_after,
        "quality_improvement": round(q_after - q_before, 1),
        "viz_payload": viz_payload
    })
