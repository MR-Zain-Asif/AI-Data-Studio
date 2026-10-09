"""Intelligent Recommendations Service with Priority Scoring."""

import uuid
import re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def generate_smart_recommendations(df: pd.DataFrame, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate rule-based data cleaning & engineering recommendations with priority scoring."""
    recommendations: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return recommendations

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # ─────────────────────────────────────────────────────────────
    # CRITICAL CATEGORY (Show first)
    # ─────────────────────────────────────────────────────────────
    # All values missing
    for col in df.columns:
        if df[col].isna().all():
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Critical",
                "title": f"Delete completely empty column '{col}'",
                "description": f"Column '{col}' has 100% missing values and carries zero information.",
                "impact_score": 10,
                "effort_score": 1,
                "priority": 10.0,
                "affected_column": str(col),
                "affected_rows": n_rows,
                "fix_function": "drop_column",
                "fix_params": {"column": str(col)},
                "before_preview": ["NaN"] * min(3, n_rows),
                "after_preview": ["<Column Removed>"],
                "confidence": 100
            })

    # Missing > 50%
    for col in df.columns:
        miss_count = int(df[col].isna().sum())
        miss_pct = (miss_count / max(n_rows, 1)) * 100
        if 50.0 < miss_pct < 100.0:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Critical",
                "title": f"High missingness in '{col}' ({miss_pct:.1f}%)",
                "description": f"Over 50% of values are missing. Delete column or impute aggressively.",
                "impact_score": 9,
                "effort_score": 2,
                "priority": 4.5,
                "affected_column": str(col),
                "affected_rows": miss_count,
                "fix_function": "drop_column",
                "fix_params": {"column": str(col)},
                "before_preview": df[col].dropna().head(2).tolist() + ["NaN"],
                "after_preview": ["<Column Removed>"],
                "confidence": 92
            })

    # Duplicate rows > 5%
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / max(n_rows, 1)) * 100
    if dup_pct >= 5.0:
        recommendations.append({
            "id": str(uuid.uuid4()),
            "category": "Critical",
            "title": f"Remove {dup_rows} duplicate rows ({dup_pct:.1f}%)",
            "description": "High duplicate row ratio degrades model generalizability and skews metrics.",
            "impact_score": 10,
            "effort_score": 1,
            "priority": 10.0,
            "affected_column": "ALL",
            "affected_rows": dup_rows,
            "fix_function": "drop_duplicates",
            "fix_params": {"keep": "first"},
            "before_preview": [f"{n_rows} total rows ({dup_rows} duplicates)"],
            "after_preview": [f"{n_rows - dup_rows} unique rows"],
            "confidence": 98
        })

    # Single unique value (Zero variance)
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1:
            val = df[col].dropna().iloc[0]
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Critical",
                "title": f"Delete constant column '{col}'",
                "description": f"Column '{col}' has only one constant value ('{val}'). It adds zero signal.",
                "impact_score": 9,
                "effort_score": 1,
                "priority": 9.0,
                "affected_column": str(col),
                "affected_rows": n_rows,
                "fix_function": "drop_column",
                "fix_params": {"column": str(col)},
                "before_preview": [str(val)] * min(3, n_rows),
                "after_preview": ["<Column Removed>"],
                "confidence": 99
            })

    # ─────────────────────────────────────────────────────────────
    # HIGH IMPACT CATEGORY
    # ─────────────────────────────────────────────────────────────
    # Numeric stored as text
    for col in object_cols:
        s = df[col].dropna().astype(str)
        if len(s) > 0:
            cleaned = s.str.replace(r'[\$,%]', '', regex=True).str.strip()
            num_converted = pd.to_numeric(cleaned, errors="coerce")
            valid_ratio = num_converted.notna().sum() / len(s)
            if valid_ratio >= 0.85:
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "category": "High",
                    "title": f"Convert text column '{col}' to numeric",
                    "description": "Numbers detected stored as string. Converting unlocks statistical aggregation and modeling.",
                    "impact_score": 8,
                    "effort_score": 1,
                    "priority": 8.0,
                    "affected_column": str(col),
                    "affected_rows": len(s),
                    "fix_function": "cast_type",
                    "fix_params": {"column": str(col), "target_type": "float"},
                    "before_preview": s.head(3).tolist(),
                    "after_preview": num_converted.dropna().head(3).tolist(),
                    "confidence": 95
                })

    # Date stored as text
    for col in object_cols:
        if any(d in str(col).lower() for d in ["date", "time", "dob", "created", "timestamp", "year"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= 0.7 * n_rows:
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "category": "High",
                    "title": f"Parse datetime in '{col}'",
                    "description": "Parsing dates unlocks time-series ordering, seasonality, and feature extraction.",
                    "impact_score": 8,
                    "effort_score": 2,
                    "priority": 4.0,
                    "affected_column": str(col),
                    "affected_rows": int(parsed.notna().sum()),
                    "fix_function": "parse_dates",
                    "fix_params": {"column": str(col), "target_format": "%Y-%m-%d"},
                    "before_preview": df[col].dropna().head(3).astype(str).tolist(),
                    "after_preview": parsed.dropna().dt.strftime("%Y-%m-%d").head(3).tolist(),
                    "confidence": 92
                })

    # Missing 10-50% -> Impute with KNN / Median
    for col in df.columns:
        miss_count = int(df[col].isna().sum())
        miss_pct = (miss_count / max(n_rows, 1)) * 100
        if 10.0 <= miss_pct <= 50.0:
            is_num = col in numeric_cols
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "High",
                "title": f"Impute missing values in '{col}' ({miss_pct:.1f}%)",
                "description": f"Impute {miss_count} missing records using {'KNN / Median' if is_num else 'Mode'} to preserve sample size.",
                "impact_score": 8,
                "effort_score": 2,
                "priority": 4.0,
                "affected_column": str(col),
                "affected_rows": miss_count,
                "fix_function": "knn_impute" if is_num else "fill_missing",
                "fix_params": {"column": str(col), "method": "knn" if is_num else "mode"},
                "before_preview": ["NaN", "...", df[col].dropna().iloc[0] if len(df[col].dropna()) else "NaN"],
                "after_preview": ["Imputed", "...", df[col].dropna().iloc[0] if len(df[col].dropna()) else "Imputed"],
                "confidence": 88
            })

    # High cardinality text (unique% > 90% in large df)
    if n_rows > 30:
        for col in object_cols:
            if df[col].nunique() / n_rows > 0.90:
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "category": "High",
                    "title": f"Drop high-cardinality ID column '{col}'",
                    "description": f"Column '{col}' is >90% unique identifiers, causing overfitting in ML models.",
                    "impact_score": 7,
                    "effort_score": 1,
                    "priority": 7.0,
                    "affected_column": str(col),
                    "affected_rows": n_rows,
                    "fix_function": "drop_column",
                    "fix_params": {"column": str(col)},
                    "before_preview": df[col].dropna().head(3).tolist(),
                    "after_preview": ["<Column Removed>"],
                    "confidence": 85
                })

    # Skewness > 2
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 10 and (s > 0).all():
            skew = float(s.skew())
            if skew > 2.0:
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "category": "High",
                    "title": f"Apply log transformation on skewed '{col}'",
                    "description": f"Skewness is {skew:.2f} (> 2.0). Log-scaling normalizes the distribution for linear/neural models.",
                    "impact_score": 7,
                    "effort_score": 2,
                    "priority": 3.5,
                    "affected_column": str(col),
                    "affected_rows": len(s),
                    "fix_function": "scale",
                    "fix_params": {"columns": [str(col)], "method": "log"},
                    "before_preview": s.head(3).round(2).tolist(),
                    "after_preview": np.log1p(s).head(3).round(2).tolist(),
                    "confidence": 90
                })

    # Outliers > 5%
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 10:
            q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q75 - q25
            if iqr > 0:
                lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
                outliers = int(((s < lower) | (s > upper)).sum())
                out_pct = (outliers / len(s)) * 100
                if out_pct >= 5.0:
                    recommendations.append({
                        "id": str(uuid.uuid4()),
                        "category": "High",
                        "title": f"Cap {outliers} outliers in '{col}' ({out_pct:.1f}%)",
                        "description": "Extreme outliers distort statistical variance and regression weights.",
                        "impact_score": 7,
                        "effort_score": 2,
                        "priority": 3.5,
                        "affected_column": str(col),
                        "affected_rows": outliers,
                        "fix_function": "winsorize",
                        "fix_params": {"column": str(col), "limits": [0.05, 0.05]},
                        "before_preview": [float(s.min()), float(s.max())],
                        "after_preview": [round(float(s.quantile(0.05)), 2), round(float(s.quantile(0.95)), 2)],
                        "confidence": 89
                    })

    # ─────────────────────────────────────────────────────────────
    # MEDIUM IMPACT CATEGORY
    # ─────────────────────────────────────────────────────────────
    # Inconsistent categories (e.g. yes, Yes, YES)
    for col in object_cols:
        s = df[col].dropna().astype(str)
        if 0 < s.nunique() <= 40:
            norm_map = {}
            for v in s.unique():
                key = str(v).strip().lower()
                norm_map.setdefault(key, []).append(str(v))
            conflicts = [variants for variants in norm_map.values() if len(variants) > 1]
            if conflicts:
                mapping = {}
                for variants in conflicts:
                    canonical = variants[0].strip().title()
                    for v in variants:
                        mapping[v] = canonical
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "category": "Medium",
                    "title": f"Standardize casing & categories in '{col}'",
                    "description": f"Found inconsistent variations: {conflicts[0][:3]}. Consolidating removes redundant categorical splits.",
                    "impact_score": 6,
                    "effort_score": 1,
                    "priority": 6.0,
                    "affected_column": str(col),
                    "affected_rows": len(s),
                    "fix_function": "standardize_categories",
                    "fix_params": {"column": str(col), "mapping": mapping},
                    "before_preview": conflicts[0][:3],
                    "after_preview": [mapping.get(c, c) for c in conflicts[0][:3]],
                    "confidence": 94
                })

    # Leading/trailing spaces
    for col in object_cols:
        s = df[col].dropna().astype(str)
        untrimmed = (s != s.str.strip()).sum()
        if untrimmed > 0:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Medium",
                "title": f"Trim whitespace in '{col}' ({untrimmed} cells)",
                "description": "Leading and trailing spaces create duplicate categories and join failures.",
                "impact_score": 5,
                "effort_score": 1,
                "priority": 5.0,
                "affected_column": str(col),
                "affected_rows": int(untrimmed),
                "fix_function": "clean_text",
                "fix_params": {"column": str(col), "operations": ["strip", "remove_extra_spaces"]},
                "before_preview": [f" '{s.iloc[0]}' "],
                "after_preview": [s.iloc[0].strip()],
                "confidence": 98
            })

    # Correlation > 0.85 between numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr.loc[c1, c2]
                if val > 0.85 and not np.isnan(val):
                    recommendations.append({
                        "id": str(uuid.uuid4()),
                        "category": "Medium",
                        "title": f"Multicollinearity between '{c1}' & '{c2}' (r={val:.2f})",
                        "description": "High collinearity causes variance inflation in regression models. Consider removing one.",
                        "impact_score": 6,
                        "effort_score": 2,
                        "priority": 3.0,
                        "affected_column": str(c2),
                        "affected_rows": n_rows,
                        "fix_function": "drop_column",
                        "fix_params": {"column": str(c2)},
                        "before_preview": [f"Correlation: {val:.2f}"],
                        "after_preview": [f"Retained '{c1}' only"],
                        "confidence": 85
                    })

    # ─────────────────────────────────────────────────────────────
    # LOW IMPACT CATEGORY (Nice-to-have)
    # ─────────────────────────────────────────────────────────────
    # Column names have spaces
    space_cols = [c for c in df.columns if " " in str(c)]
    if space_cols:
        recommendations.append({
            "id": str(uuid.uuid4()),
            "category": "Low",
            "title": "Rename columns to replace spaces with underscores",
            "description": f"{len(space_cols)} column names have spaces. Snake_case naming prevents syntax issues in SQL and Pandas.",
            "impact_score": 3,
            "effort_score": 1,
            "priority": 3.0,
            "affected_column": ", ".join(space_cols[:3]),
            "affected_rows": len(space_cols),
            "fix_function": "rename_columns",
            "fix_params": {c: c.replace(" ", "_") for c in space_cols},
            "before_preview": space_cols[:3],
            "after_preview": [c.replace(" ", "_") for c in space_cols[:3]],
            "confidence": 99
        })

    # Boolean stored as 0/1
    for col in numeric_cols:
        vals = set(df[col].dropna().unique())
        if vals == {0, 1} or vals == {0} or vals == {1}:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Low",
                "title": f"Convert binary column '{col}' (0/1) to Boolean",
                "description": "Explicit boolean type clarifies semantic meaning and improves dashboard displays.",
                "impact_score": 3,
                "effort_score": 1,
                "priority": 3.0,
                "affected_column": str(col),
                "affected_rows": n_rows,
                "fix_function": "cast_type",
                "fix_params": {"column": str(col), "target_type": "bool"},
                "before_preview": [0, 1, 0],
                "after_preview": [False, True, False],
                "confidence": 95
            })

    # Sort by priority descending
    # Maintain Category order: Critical -> High -> Medium -> Low
    category_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recommendations.sort(
        key=lambda r: (category_order.get(r["category"], 99), -r["priority"])
    )

    return recommendations


def apply_critical_recommendations(df: pd.DataFrame) -> (pd.DataFrame, int, List[str]):
    """Automatically apply all Critical recommendations."""
    new_df = df.copy()
    recs = generate_smart_recommendations(new_df)
    critical_recs = [r for r in recs if r["category"] == "Critical"]
    applied_logs = []
    affected_rows = 0

    for r in critical_recs:
        fn = r.get("fix_function")
        params = r.get("fix_params", {})

        if fn == "drop_column":
            col = params.get("column")
            if col in new_df.columns:
                new_df = new_df.drop(columns=[col])
                applied_logs.append(f"Dropped column '{col}'")
                affected_rows += len(new_df)

        elif fn == "drop_duplicates":
            before = len(new_df)
            new_df = new_df.drop_duplicates(keep=params.get("keep", "first"))
            removed = before - len(new_df)
            applied_logs.append(f"Removed {removed} duplicate rows")
            affected_rows += removed

    return new_df, affected_rows, applied_logs
