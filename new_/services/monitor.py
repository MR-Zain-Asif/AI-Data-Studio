"""Real-time Data Quality Monitor Service."""

import re
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def _sanitize_num(val, default=0.0):
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return float(val)


def calculate_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive quality metrics across 6 dimensions."""
    if df is None or df.empty:
        return {
            "overall_score": 0.0,
            "grade": "F",
            "completeness": {"score": 0.0, "missing_cells": 0, "missing_percentage": 0.0, "columns_with_missing": []},
            "consistency": {"score": 0.0, "type_mismatches": 0, "format_issues": 0, "inconsistent_categories": []},
            "accuracy": {"score": 0.0, "outlier_count": 0, "outlier_percentage": 0.0, "suspicious_values": []},
            "uniqueness": {"score": 0.0, "duplicate_rows": 0, "duplicate_percentage": 0.0, "duplicate_columns": []},
            "validity": {"score": 0.0, "invalid_emails": 0, "invalid_phones": 0, "invalid_dates": 0, "custom_rule_failures": 0},
            "timeliness": {"score": 100.0, "date_columns_detected": [], "oldest_date": None, "newest_date": None, "date_gaps_detected": False},
            "improvements": []
        }

    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)

    # 1. Completeness
    missing_by_col = df.isna().sum()
    missing_cells = int(missing_by_col.sum())
    missing_percentage = round((missing_cells / total_cells) * 100, 2)
    columns_with_missing = []
    for col, count in missing_by_col.items():
        if count > 0:
            columns_with_missing.append({
                "column": str(col),
                "missing_count": int(count),
                "missing_pct": round((count / max(n_rows, 1)) * 100, 2)
            })
    completeness_score = max(0.0, min(100.0, round(100.0 - (missing_percentage * 1.5), 2)))

    # 2. Uniqueness
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_rows / max(n_rows, 1)) * 100, 2)
    duplicate_columns = []
    col_names = list(df.columns)
    for i in range(len(col_names)):
        for j in range(i + 1, len(col_names)):
            c1, c2 = col_names[i], col_names[j]
            try:
                if df[c1].equals(df[c2]):
                    duplicate_columns.append(f"{c1} == {c2}")
            except Exception:
                pass
    uniqueness_score = max(0.0, min(100.0, round(100.0 - (duplicate_percentage * 2.0) - (len(duplicate_columns) * 10.0), 2)))

    # 3. Accuracy & Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_numeric_cells = max(len(numeric_cols) * n_rows, 1)
    total_outliers = 0
    suspicious_values = []

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 4:
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                outliers = int(((s < lower) | (s > upper)).sum())
                total_outliers += outliers

        # Suspicious negatives
        col_lower = str(col).lower()
        if any(w in col_lower for w in ["age", "price", "salary", "cost", "revenue", "count", "quantity", "qty"]):
            negs = int((s < 0).sum())
            if negs > 0:
                suspicious_values.append(f"Negative values ({negs}) in positive column '{col}'")
        if any(w in col_lower for w in ["percent", "pct", "rate", "probability"]):
            over100 = int((s > 100).sum())
            if over100 > 0:
                suspicious_values.append(f"Values > 100 ({over100}) in percentage column '{col}'")

    outlier_percentage = round((total_outliers / total_numeric_cells) * 100, 2) if numeric_cols else 0.0
    accuracy_score = max(0.0, min(100.0, round(100.0 - min(40.0, outlier_percentage * 1.5) - min(40.0, len(suspicious_values) * 8.0), 2)))

    # 4. Consistency
    type_mismatches = 0
    format_issues = 0
    inconsistent_categories = []
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in object_cols:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        # Check numeric stored as string
        cleaned_num = s.str.replace(r'[\$,%]', '', regex=True).str.strip()
        num_converted = pd.to_numeric(cleaned_num, errors="coerce")
        valid_num_count = int(num_converted.notna().sum())
        if valid_num_count > 0.8 * len(s) and valid_num_count < len(s):
            type_mismatches += 1
        elif valid_num_count == len(s) and len(s) > 0:
            type_mismatches += 1  # completely numeric stored as string

        # Inconsistent categories (e.g. Yes, YES, yes)
        unique_vals = s.unique()
        if len(unique_vals) <= 50:
            norm_map: Dict[str, List[str]] = {}
            for v in unique_vals:
                key = str(v).strip().lower()
                norm_map.setdefault(key, []).append(str(v))
            inconsistencies = [variants for variants in norm_map.values() if len(variants) > 1]
            if inconsistencies:
                inconsistent_categories.append({
                    "column": str(col),
                    "conflicts": [list(item) for item in inconsistencies[:3]]
                })

        # Whitespace format issues
        has_spaces = (s != s.str.strip()).sum()
        if has_spaces > 0:
            format_issues += 1

    consistency_score = max(0.0, min(100.0, round(100.0 - (type_mismatches * 10.0) - (len(inconsistent_categories) * 10.0) - (format_issues * 5.0), 2)))

    # 5. Validity
    email_regex = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone_regex = re.compile(r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$")
    invalid_emails = 0
    invalid_phones = 0
    invalid_dates = 0
    custom_rule_failures = len(suspicious_values)

    for col in object_cols:
        col_l = str(col).lower()
        s = df[col].dropna().astype(str)
        if "email" in col_l or s.str.contains("@").sum() > 0.5 * len(s):
            invalid_emails += int((~s.str.match(email_regex)).sum())
        elif any(p in col_l for p in ["phone", "tel", "mobile"]):
            clean_phones = s.str.replace(r'[\s\(\)\-\.]', '', regex=True)
            invalid_phones += int((~clean_phones.str.isdigit()).sum())

    validity_penalty = min(50.0, invalid_emails * 3.0) + min(30.0, invalid_phones * 3.0) + min(20.0, custom_rule_failures * 5.0)
    validity_score = max(0.0, min(100.0, round(100.0 - validity_penalty, 2)))

    # 6. Timeliness
    date_cols = []
    oldest_date = None
    newest_date = None
    date_gaps_detected = False

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(str(col))
        elif any(d in str(col).lower() for d in ["date", "time", "timestamp", "year"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0.6 * len(df):
                    date_cols.append(str(col))
            except Exception:
                pass

    timeliness_score = 100.0
    if date_cols:
        try:
            sample_col = date_cols[0]
            dates = pd.to_datetime(df[sample_col], errors="coerce").dropna().sort_values()
            if len(dates) > 0:
                oldest_date = str(dates.iloc[0].date())
                newest_date = str(dates.iloc[-1].date())
                diffs = dates.diff().dt.days
                if (diffs > 365).any():
                    date_gaps_detected = True
                    timeliness_score -= 15.0
        except Exception:
            pass

    # Overall Score formula
    overall = (
        completeness_score * 0.30 +
        consistency_score * 0.20 +
        accuracy_score * 0.20 +
        uniqueness_score * 0.15 +
        validity_score * 0.10 +
        timeliness_score * 0.05
    )
    overall_score = round(max(0.0, min(100.0, overall)), 1)

    # Grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    elif overall_score >= 60:
        grade = "D"
    else:
        grade = "F"

    # Improvements ranked by impact
    improvements = []
    if missing_cells > 0:
        top_miss = columns_with_missing[0]["column"] if columns_with_missing else "columns"
        improvements.append({
            "action": f"Impute or drop missing values in '{top_miss}' and others ({missing_cells} cells)",
            "impact": min(25, int(missing_percentage * 0.8) + 5),
            "difficulty": "easy"
        })
    if duplicate_rows > 0:
        improvements.append({
            "action": f"Remove {duplicate_rows} duplicate rows ({duplicate_percentage}%)",
            "impact": min(20, int(duplicate_percentage * 1.5) + 5),
            "difficulty": "easy"
        })
    if total_outliers > 0:
        improvements.append({
            "action": f"Handle {total_outliers} statistical outliers using Winsorization or IQR capping",
            "impact": min(18, int(outlier_percentage * 1.2) + 4),
            "difficulty": "medium"
        })
    if type_mismatches > 0:
        improvements.append({
            "action": f"Fix data type mismatches across {type_mismatches} columns (numeric stored as text)",
            "impact": 12,
            "difficulty": "easy"
        })
    if inconsistent_categories:
        col_names_cat = ", ".join(f"'{c['column']}'" for c in inconsistent_categories[:2])
        improvements.append({
            "action": f"Standardize text casing & categorical variations in {col_names_cat}",
            "impact": 10,
            "difficulty": "easy"
        })
    if suspicious_values:
        improvements.append({
            "action": f"Resolve domain constraint violations ({suspicious_values[0]})",
            "impact": 15,
            "difficulty": "medium"
        })
    if invalid_emails > 0:
        improvements.append({
            "action": f"Clean or drop {invalid_emails} invalid email addresses",
            "impact": 8,
            "difficulty": "easy"
        })

    # Sort improvements by impact descending
    improvements.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "overall_score": overall_score,
        "grade": grade,
        "completeness": {
            "score": completeness_score,
            "missing_cells": missing_cells,
            "missing_percentage": missing_percentage,
            "columns_with_missing": columns_with_missing
        },
        "consistency": {
            "score": consistency_score,
            "type_mismatches": type_mismatches,
            "format_issues": format_issues,
            "inconsistent_categories": inconsistent_categories
        },
        "accuracy": {
            "score": accuracy_score,
            "outlier_count": total_outliers,
            "outlier_percentage": outlier_percentage,
            "suspicious_values": suspicious_values
        },
        "uniqueness": {
            "score": uniqueness_score,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "duplicate_columns": duplicate_columns
        },
        "validity": {
            "score": validity_score,
            "invalid_emails": invalid_emails,
            "invalid_phones": invalid_phones,
            "invalid_dates": invalid_dates,
            "custom_rule_failures": custom_rule_failures
        },
        "timeliness": {
            "score": timeliness_score,
            "date_columns_detected": date_cols,
            "oldest_date": oldest_date,
            "newest_date": newest_date,
            "date_gaps_detected": date_gaps_detected
        },
        "improvements": improvements
    }
