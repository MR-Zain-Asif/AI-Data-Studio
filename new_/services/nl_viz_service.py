"""Natural Language to Visualization (NL-to-Chart) Service.
Translates plain language prompts into Chart.js data objects with auto-aggregation and color palettes.
"""

import os
import re
import json
import math
import httpx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

VIBRANT_PALETTE = [
    "#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", 
    "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#84cc16"
]


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


def generate_nl_chart(prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze prompt and dataset, then construct Chart.js ready payload."""
    if df is None or df.empty:
        return {"error": "Dataset is empty"}

    p_low = prompt.lower()
    cols = list(df.columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    # 1. Match columns mentioned in prompt
    matched_cols = [c for c in cols if c.lower() in p_low]
    matched_num = [c for c in matched_cols if c in num_cols]
    matched_cat = [c for c in matched_cols if c in cat_cols]

    # Chart type detection
    chart_type = "bar"
    if any(k in p_low for k in ["line", "trend", "over time", "growth"]):
        chart_type = "line"
    elif any(k in p_low for k in ["pie", "donut", "doughnut", "share", "proportion"]):
        chart_type = "doughnut" if "donut" in p_low or "doughnut" in p_low else "pie"
    elif any(k in p_low for k in ["scatter", "correlation", "relationship", "vs"]):
        chart_type = "scatter"
    elif any(k in p_low for k in ["hist", "distribution", "spread"]):
        chart_type = "bar"

    # Default selections if not matched
    x_col = matched_cat[0] if matched_cat else (matched_cols[0] if matched_cols else (cat_cols[0] if cat_cols else cols[0]))
    y_col = matched_num[0] if matched_num else (num_cols[0] if num_cols else None)

    # If scatter plot with 2 numeric cols
    if chart_type == "scatter" and len(matched_num) >= 2:
        x_col = matched_num[0]
        y_col = matched_num[1]
    elif chart_type == "scatter" and len(num_cols) >= 2:
        x_col = num_cols[0]
        y_col = num_cols[1]

    title = f"{chart_type.upper()}: {x_col}" + (f" vs {y_col}" if y_col and y_col != x_col else "")

    # Build chart data structures
    labels = []
    data_points = []

    try:
        if chart_type == "scatter" and y_col:
            sample_df = df[[x_col, y_col]].dropna().head(100)
            scatter_data = [{"x": float(row[x_col]), "y": float(row[y_col])} for _, row in sample_df.iterrows()]
            return _sanitize({
                "chart_type": "scatter",
                "title": title,
                "x_col": x_col,
                "y_col": y_col,
                "labels": [f"Pt {i+1}" for i in range(len(scatter_data))],
                "datasets": [{
                    "label": f"{y_col} vs {x_col}",
                    "data": scatter_data,
                    "backgroundColor": "#2563eb",
                    "borderColor": "#1d4ed8"
                }]
            })

        elif y_col and x_col != y_col and x_col in cat_cols:
            # Grouped aggregation (e.g. mean or sum of Y grouped by X)
            grouped = df.groupby(x_col)[y_col].agg(["mean", "count"]).reset_index().head(15)
            labels = [str(val) for val in grouped[x_col].tolist()]
            data_points = [round(float(val), 2) for val in grouped["mean"].tolist()]
            metric_name = f"Average {y_col}"

        elif y_col and x_col == y_col:
            # Numerical histogram / frequency bins
            counts, bin_edges = np.histogram(df[y_col].dropna(), bins=10)
            labels = [f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}" for i in range(len(counts))]
            data_points = [int(c) for c in counts]
            metric_name = f"{y_col} Frequency"

        else:
            # Categorical value counts
            val_counts = df[x_col].value_counts().head(12)
            labels = [str(k) for k in val_counts.index.tolist()]
            data_points = [int(v) for v in val_counts.values.tolist()]
            metric_name = f"{x_col} Count"

    except Exception:
        # Fallback to simple top 10 rows
        labels = [f"Row {i+1}" for i in range(min(10, len(df)))]
        data_points = [1] * len(labels)
        metric_name = "Count"

    colors = VIBRANT_PALETTE if chart_type in ["pie", "doughnut"] else "#2563eb"
    border_colors = VIBRANT_PALETTE if chart_type in ["pie", "doughnut"] else "#1d4ed8"

    return _sanitize({
        "chart_type": chart_type,
        "title": title,
        "x_col": x_col,
        "y_col": y_col,
        "labels": labels,
        "datasets": [{
            "label": metric_name,
            "data": data_points,
            "backgroundColor": colors,
            "borderColor": border_colors,
            "borderWidth": 1.5
        }]
    })
