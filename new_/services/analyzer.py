"""Data analysis engine — all logic in pure pandas and numpy."""

import pandas as pd
import numpy as np
import re
import difflib
from typing import Any
from dateutil import parser as date_parser

# Values that should be treated as missing (case-insensitive)
MISSING_SENTINELS = {"", "null", "na", "n/a", "none", "nan", "nil", "missing", "-", "--", "n.a.", "n.a", "#n/a", "#na", "#null"}


class DataAnalyzer:
    """Performs comprehensive data quality analysis on a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, dtypes=None, outliers=None, missing=None):
        self.df = df
        self._cached_missing = missing
        self._cached_duplicates = None
        self._cached_outliers = outliers
        self._cached_empty = None
        self._cached_constant = None
        self._cached_inconsistent_text = None
        self._cached_whitespace = None
        self._cached_date_inconsistency = None
        self._cached_dtypes = dtypes

    # ── Missing values ───────────────────────────────────────────────
    def missing_values(self) -> dict[str, Any]:
        """Count missing values per column including sentinel strings."""
        if self._cached_missing is not None:
            return self._cached_missing

        result = {}
        for col in self.df.columns:
            native_missing = self.df[col].isna()

            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                unique_vals = self.df[col].dropna().unique()
                sentinel_uniques = [val for val in unique_vals if str(val).strip().lower() in MISSING_SENTINELS or str(val).strip() == '']
                if sentinel_uniques:
                    if len(sentinel_uniques) == 1:
                        sentinel_mask = (self.df[col] == sentinel_uniques[0])
                    else:
                        sentinel_mask = self.df[col].isin(sentinel_uniques)
                    combined_mask = native_missing | sentinel_mask
                else:
                    combined_mask = native_missing
            else:
                combined_mask = native_missing

            missing_count = int(combined_mask.sum())
            if missing_count > 0:
                result[col] = {
                    "count": missing_count,
                    "percentage": round(missing_count / len(self.df) * 100, 2),
                    "rows": self.df.index[combined_mask][:100].tolist(),
                }
        self._cached_missing = result
        return result

    # ── Duplicate rows ───────────────────────────────────────────────
    def duplicate_rows(self) -> dict[str, Any]:
        """Find exact duplicate rows and return their indices."""
        if self._cached_duplicates is not None:
            return self._cached_duplicates

        mask = self.df.duplicated(keep="first")
        dup_count = int(mask.sum())
        dup_indices = self.df.index[mask][:200].tolist() if dup_count > 0 else []
        result = {
            "count": dup_count,
            "indices": dup_indices,
        }
        self._cached_duplicates = result
        return result

    # ── Data type detection ──────────────────────────────────────────
    def data_type_detection(self) -> dict[str, str]:
        """Detect semantic data type per column: number, text, date, boolean, mixed."""
        if self._cached_dtypes is not None:
            return self._cached_dtypes

        result = {}
        for col in self.df.columns:
            series = self.df[col].dropna()
            if series.empty:
                result[col] = "empty"
                continue

            if pd.api.types.is_bool_dtype(series):
                result[col] = "boolean"
                continue

            if pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
                    unique_vals = set(series.unique())
                    if unique_vals.issubset({0, 1}):
                        result[col] = "boolean"
                        continue
                result[col] = "number"
                continue

            unique_raw = series.unique()
            sentinel_uniques = [val for val in unique_raw if str(val).strip().lower() in MISSING_SENTINELS or str(val).strip() == '']
            if sentinel_uniques:
                series = series[~series.isin(sentinel_uniques)]
                if series.empty:
                    result[col] = "empty"
                    continue
                unique_raw = [v for v in unique_raw if v not in sentinel_uniques]

            is_bool = False
            if len(unique_raw) <= 10:
                unique_vals = {str(x).strip().lower() for x in unique_raw}
                if unique_vals.issubset({"true", "false", "yes", "no", "0", "1"}):
                    is_bool = True

            if is_bool:
                result[col] = "boolean"
                continue

            sample_series = series.head(1000)
            sample_coerced = pd.to_numeric(sample_series, errors="coerce")
            sample_ratio = sample_coerced.notna().sum() / len(sample_series) if len(sample_series) > 0 else 0
            if sample_ratio < 0.5:
                result[col] = "text"
                continue

            numeric_coerced = pd.to_numeric(series, errors="coerce")
            numeric_ratio = numeric_coerced.notna().sum() / len(series)
            if numeric_ratio > 0.8:
                result[col] = "number" if numeric_ratio > 0.95 else "mixed"
                continue

            sample = series.head(20)
            date_count = 0
            for val in sample:
                val_str = str(val).strip()
                if not any(c.isdigit() for c in val_str):
                    continue
                if val_str.replace('.', '', 1).isdigit():
                    if len(val_str) != 8:
                        continue
                try:
                    date_parser.parse(val_str, fuzzy=False)
                    date_count += 1
                except (ValueError, OverflowError, TypeError):
                    pass
            if len(sample) > 0 and date_count / len(sample) > 0.7:
                result[col] = "date"
                continue

            result[col] = "text"
        self._cached_dtypes = result
        return result

    # ── Outlier detection (IQR) ──────────────────────────────────────
    def outlier_detection(self) -> dict[str, Any]:
        """IQR-based outlier detection for every numeric column."""
        if self._cached_outliers is not None:
            return self._cached_outliers

        result = {}
        dtypes = self.data_type_detection()
        for col in self.df.columns:
            if pd.api.types.is_bool_dtype(self.df[col]) or dtypes.get(col) == "boolean":
                continue
            if not (pd.api.types.is_numeric_dtype(self.df[col]) or dtypes.get(col) == "number"):
                continue

            if pd.api.types.is_float_dtype(self.df[col]):
                coerced_col = self.df[col]
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                coerced_col = self.df[col].astype(float)
            else:
                coerced_col = pd.to_numeric(self.df[col], errors="coerce").astype(float)

            series = coerced_col.dropna()
            if series.empty:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                tol = 1e-9 * iqr
            else:
                median_val = float(series.median())
                mad = float((series - median_val).abs().median())
                if mad > 0:
                    lower = median_val - 3.0 * 1.4826 * mad
                    upper = median_val + 3.0 * 1.4826 * mad
                    tol = 1e-9 * mad
                else:
                    if series.nunique() > 1:
                        lower = median_val
                        upper = median_val
                        tol = 1e-9
                    else:
                        continue

            outlier_mask = (coerced_col < lower - tol) | (coerced_col > upper + tol)
            outlier_mask = outlier_mask & coerced_col.notna()
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                result[col] = {
                    "count": outlier_count,
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "indices": self.df.index[outlier_mask][:200].tolist(),
                }
        self._cached_outliers = result
        return result

    # ── Constant columns ─────────────────────────────────────────────
    def constant_columns(self) -> list[str]:
        """Columns where all non-null values are identical."""
        if self._cached_constant is not None:
            return self._cached_constant

        result = []
        for col in self.df.columns:
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                first_val = non_null.iloc[0]
                if (non_null == first_val).all():
                    result.append(col)
        self._cached_constant = result
        return result

    # ── Empty columns (80%+ missing) ─────────────────────────────────
    def empty_columns(self) -> list[str]:
        """Columns with 80%+ missing values (including sentinels)."""
        if self._cached_empty is not None:
            return self._cached_empty

        result = []
        missing = self.missing_values()
        for col in self.df.columns:
            if col in missing:
                if missing[col]["percentage"] >= 80:
                    result.append(col)
            else:
                pct = self.df[col].isna().sum() / len(self.df) * 100 if len(self.df) > 0 else 0
                if pct >= 80:
                    result.append(col)
        self._cached_empty = result
        return result

    # ── Inconsistent text (mixed case) ───────────────────────────────
    def inconsistent_text(self) -> dict[str, Any]:
        """Find columns with mixed casing."""
        if self._cached_inconsistent_text is not None:
            return self._cached_inconsistent_text

        result = {}
        text_cols = [c for c in self.df.columns if pd.api.types.is_string_dtype(self.df[c]) or pd.api.types.is_object_dtype(self.df[c])]
        for col in text_cols:
            unique_raw = self.df[col].dropna().unique()
            if len(unique_raw) < 2:
                continue

            unique_series = pd.Series(unique_raw).astype(str)
            unique_series = unique_series[~unique_series.str.strip().str.lower().isin(MISSING_SENTINELS)]
            if len(unique_series) < 2:
                continue

            if len(unique_series) > 10000:
                unique_series = unique_series.iloc[:10000]

            has_upper = unique_series.str.isupper().any()
            has_lower = unique_series.str.islower().any()
            has_title = unique_series.str.istitle().any()
            cases_present = sum([has_upper, has_lower, has_title])
            if cases_present >= 2:
                result[col] = {
                    "has_upper": bool(has_upper),
                    "has_lower": bool(has_lower),
                    "has_title": bool(has_title),
                    "sample_values": unique_series.head(5).tolist(),
                }
        self._cached_inconsistent_text = result
        return result

    # ── Whitespace issues ────────────────────────────────────────────
    def whitespace_issues(self) -> dict[str, Any]:
        """Find text cells with leading or trailing whitespace."""
        if self._cached_whitespace is not None:
            return self._cached_whitespace

        result = {}
        text_cols = [c for c in self.df.columns if pd.api.types.is_string_dtype(self.df[c]) or pd.api.types.is_object_dtype(self.df[c])]
        for col in text_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue
            unique_raw = series.unique()
            ws_uniques = [val for val in unique_raw if str(val).strip() != str(val)]
            if ws_uniques:
                if len(ws_uniques) == 1:
                    ws_count = int((self.df[col] == ws_uniques[0]).sum())
                else:
                    ws_count = int(self.df[col].isin(ws_uniques).sum())
                if ws_count > 0:
                    result[col] = {
                        "count": ws_count,
                        "percentage": round(ws_count / len(self.df) * 100, 2),
                    }
        self._cached_whitespace = result
        return result

    # ── Duplicate column names ───────────────────────────────────────
    def duplicate_columns(self) -> list[str]:
        """Find columns with identical names."""
        cols = list(self.df.columns)
        seen = set()
        duplicates = []
        for c in cols:
            if c in seen:
                duplicates.append(c)
            seen.add(c)
        return duplicates

    # ── Date inconsistency ───────────────────────────────────────────
    def date_inconsistency(self) -> dict[str, Any]:
        """Find columns that look like dates but have mixed formats."""
        if self._cached_date_inconsistency is not None:
            return self._cached_date_inconsistency

        result = {}
        dtypes = self.data_type_detection()
        for col, dtype in dtypes.items():
            if dtype != "date":
                continue

            non_null = self.df[col].dropna()
            head_str = non_null.head(30).astype(str)
            formats_found = set()
            for val in head_str:
                val_str = val.strip()
                if "/" in val_str:
                    formats_found.add("slash")
                if "-" in val_str:
                    formats_found.add("dash")
                if "." in val_str and not val_str.replace(".", "").isdigit():
                    formats_found.add("dot")
            if len(formats_found) > 1:
                result[col] = {
                    "formats_found": list(formats_found),
                    "sample_values": head_str.head(5).tolist(),
                }
        self._cached_date_inconsistency = result
        return result

    # ── Quality score ────────────────────────────────────────────────
    def quality_score(self, issues: list[dict] | None = None) -> float:
        """Calculate 0-100 quality score using proportional formula."""
        if self.df.empty:
            return 0.0

        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        total_cells = total_rows * total_cols
        if total_cells == 0:
            return 0.0

        missing_info = self.missing_values()
        total_missing = sum(info["count"] for info in missing_info.values())

        dups = self.duplicate_rows()
        dup_count = dups["count"]

        outliers = self.outlier_detection()
        total_outlier_cells = sum(info["count"] for info in outliers.values())

        other_issue_count = 0
        other_issue_count += len(self.inconsistent_text())
        other_issue_count += len(self.whitespace_issues())
        other_issue_count += len(self.empty_columns())
        other_issue_count += len(self.constant_columns())
        other_issue_count += len(self.duplicate_columns())
        other_issue_count += len(self.date_inconsistency())

        missing_penalty = (total_missing / total_cells) * 40 if total_cells > 0 else 0
        duplicate_penalty = (dup_count / total_rows) * 20 if total_rows > 0 else 0
        outlier_penalty = (total_outlier_cells / total_rows) * 100 if total_rows > 0 else 0
        other_penalty = other_issue_count * 1

        score = 100.0 - missing_penalty - duplicate_penalty - outlier_penalty - other_penalty
        return round(max(0.0, min(100.0, score)), 2)

    # ── Build issues list ────────────────────────────────────────────
    def _build_issues(self) -> list[dict]:
        """Build the list of all issues found."""
        issues: list[dict] = []

        missing = self.missing_values()
        for col, info in missing.items():
            severity = "critical" if info["percentage"] > 50 else ("warning" if info["percentage"] > 10 else "info")
            issues.append({
                "issue_type": "missing_values",
                "severity": severity,
                "column": col,
                "description": f"{info['count']} missing values ({info['percentage']}%)",
                "affected_count": info["count"],
                "recommendation": f"Fill missing values in '{col}' using mean (numbers) or mode (text)",
                "operation": "fill_missing",
                "method": "mean",
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        dups = self.duplicate_rows()
        if dups["count"] > 0:
            severity = "critical" if dups["count"] >= len(self.df) * 0.1 else "warning"
            issues.append({
                "issue_type": "duplicate_rows",
                "severity": severity,
                "column": None,
                "description": f"{dups['count']} duplicate rows found",
                "affected_count": dups["count"],
                "recommendation": "Remove duplicate rows to eliminate redundant data",
                "operation": "remove_duplicates",
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        outliers = self.outlier_detection()
        for col, info in outliers.items():
            issues.append({
                "issue_type": "outliers",
                "severity": "warning",
                "column": col,
                "description": f"{info['count']} outliers detected (bounds: {info['lower_bound']} – {info['upper_bound']})",
                "affected_count": info["count"],
                "recommendation": f"Cap or remove outliers in '{col}'",
                "operation": "fix_outliers",
                "method": "cap",
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        empty = self.empty_columns()
        for col in empty:
            issues.append({
                "issue_type": "empty_columns",
                "severity": "critical",
                "column": col,
                "description": f"Column '{col}' has 80%+ missing values",
                "affected_count": 1,
                "recommendation": f"Remove nearly-empty column '{col}'",
                "operation": "remove_empty_columns",
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        const = self.constant_columns()
        for col in const:
            issues.append({
                "issue_type": "constant_columns",
                "severity": "info",
                "column": col,
                "description": f"Column '{col}' has only one unique value",
                "affected_count": 1,
                "recommendation": f"Consider removing constant column '{col}'",
                "operation": "remove_constant_columns",
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        text_issues = self.inconsistent_text()
        for col, info in text_issues.items():
            issues.append({
                "issue_type": "inconsistent_text",
                "severity": "warning",
                "column": col,
                "description": f"Mixed casing detected in '{col}' (e.g., {', '.join(info['sample_values'][:3])})",
                "affected_count": len(self.df[col].dropna()),
                "recommendation": f"Standardize text case in '{col}'",
                "operation": "standardize_case",
                "method": "lower",
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        ws = self.whitespace_issues()
        for col, info in ws.items():
            issues.append({
                "issue_type": "whitespace",
                "severity": "warning",
                "column": col,
                "description": f"{info['count']} cells with leading/trailing whitespace ({info['percentage']}%)",
                "affected_count": info["count"],
                "recommendation": f"Trim whitespace in '{col}'",
                "operation": "trim_whitespace",
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        dup_cols = self.duplicate_columns()
        if dup_cols:
            issues.append({
                "issue_type": "duplicate_columns",
                "severity": "warning",
                "column": None,
                "description": f"Duplicate column names found: {', '.join(dup_cols)}",
                "affected_count": len(dup_cols),
                "recommendation": "Rename or remove duplicate column names",
                "operation": None,
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        date_issues = self.date_inconsistency()
        for col, info in date_issues.items():
            issues.append({
                "issue_type": "date_inconsistency",
                "severity": "warning",
                "column": col,
                "description": f"Mixed date formats in '{col}' ({', '.join(info['formats_found'])})",
                "affected_count": len(self.df[col].dropna()),
                "recommendation": f"Standardize date format in '{col}'",
                "operation": "convert_dates",
                "method": None,
                "value": None,
                "new_name": None,
                "case": None,
                "dtype": None,
                "date_format": None
            })

        return issues

    # ── Full report ──────────────────────────────────────────────────
    def full_report(self) -> dict[str, Any]:
        """Run all checks and return a combined report."""
        issues = self._build_issues()

        recommendations: list[str] = []
        missing = self.missing_values()
        dups = self.duplicate_rows()
        outliers = self.outlier_detection()
        empty = self.empty_columns()
        const = self.constant_columns()
        text_issues = self.inconsistent_text()
        ws = self.whitespace_issues()

        if missing:
            recommendations.append("Fill or remove missing values to improve data completeness")
        if dups["count"] > 0:
            recommendations.append("Remove duplicate rows")
        if outliers:
            recommendations.append("Review and handle outlier values")
        if empty:
            recommendations.append("Remove columns that are mostly empty")
        if text_issues:
            recommendations.append("Standardize text casing in affected columns")
        if ws:
            recommendations.append("Trim whitespace from text columns")

        score = self.quality_score(issues)

        total_outlier_cells = sum(info["count"] for info in outliers.values())
        total_missing_cells = sum(info["count"] for info in missing.values())

        summary = {
            "missing_columns": len(missing),
            "missing_cells": total_missing_cells,
            "duplicate_rows": dups["count"],
            "outlier_columns": len(outliers),
            "outlier_cells": total_outlier_cells,
            "text_issues": len(text_issues) + len(ws),
            "empty_columns": len(empty),
            "constant_columns": len(const),
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "total_cells": len(self.df) * len(self.df.columns),
        }

        return {
            "quality_score": score,
            "total_issues": len(issues),
            "issues": issues,
            "recommendations": recommendations,
            "summary": summary,
        }

    # ── Per-column statistics ────────────────────────────────────────
    def column_statistics(self) -> list[dict[str, Any]]:
        """Return detailed statistics for each column."""
        dtypes = self.data_type_detection()
        outliers = self.outlier_detection()
        stats = []
        for col in self.df.columns:
            series = self.df[col]
            total = len(series)

            missing_info = self.missing_values()
            missing = missing_info.get(col, {}).get("count", 0)

            non_null = series.dropna()

            if pd.api.types.is_numeric_dtype(series):
                unique_count = int(non_null.nunique())
            else:
                unique_count = None

            stat: dict[str, Any] = {
                "column_name": col,
                "data_type": dtypes.get(col, "unknown"),
                "total_count": total,
                "missing_count": missing,
                "missing_percentage": round(missing / max(total, 1) * 100, 2),
                "unique_count": unique_count,
                "outlier_count": outliers[col]["count"] if col in outliers else 0,
            }

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                stat["min_value"] = float(non_null.min()) if not non_null.empty else None
                stat["max_value"] = float(non_null.max()) if not non_null.empty else None
                stat["mean_value"] = round(float(non_null.mean()), 4) if not non_null.empty else None
                stat["median_value"] = round(float(non_null.median()), 4) if not non_null.empty else None
                stat["std_value"] = round(float(non_null.std()), 4) if not non_null.empty and len(non_null) > 1 else None
            else:
                stat["min_value"] = None
                stat["max_value"] = None
                stat["mean_value"] = None
                stat["median_value"] = None
                stat["std_value"] = None

            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or dtypes.get(col) == "text":
                str_series = non_null.astype(str)
                unique_str = str_series.unique()
                stat["unique_count"] = len(unique_str)

                sentinel_str = [x for x in unique_str if str(x).strip().lower() in MISSING_SENTINELS]
                if sentinel_str:
                    str_series = str_series[~str_series.isin(sentinel_str)]

                if not str_series.empty:
                    val_counts = str_series.value_counts()
                    stat["most_common"] = str(val_counts.index[0]) if not val_counts.empty else None

                    unique_clean = [x for x in unique_str if x not in sentinel_str]
                    if len(unique_clean) > 0:
                        lengths = np.vectorize(len)(unique_clean)
                        min_idx = np.argmin(lengths)
                        max_idx = np.argmax(lengths)
                        stat["shortest_value"] = str(unique_clean[min_idx])
                        stat["longest_value"] = str(unique_clean[max_idx])
                    else:
                        stat["shortest_value"] = None
                        stat["longest_value"] = None
                else:
                    stat["most_common"] = None
                    stat["shortest_value"] = None
                    stat["longest_value"] = None
            else:
                stat["most_common"] = None
                stat["shortest_value"] = None
                stat["longest_value"] = None

            stats.append(stat)
        return stats

    # ── FEATURE 3: Data Validation Rules Engine ──────────────────────
    @staticmethod
    def validate_rules(df: pd.DataFrame, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate dataset against user-defined rules."""
        results = []
        total_rows = len(df)

        for rule in rules:
            col = rule.get("column")
            rtype = rule.get("rule_type")
            val = rule.get("value")
            name = rule.get("friendly_name") or f"{col} ({rtype})"

            if col not in df.columns:
                results.append({
                    "rule_name": name,
                    "column": col,
                    "rule_type": rtype,
                    "passed": False,
                    "failed_count": total_rows,
                    "pass_percentage": 0.0,
                    "failed_row_indices": list(range(min(total_rows, 100))),
                    "message": f"Column '{col}' does not exist"
                })
                continue

            series = df[col]
            failed_indices = []

            if rtype == "min_value":
                limit = float(val) if val is not None else 0
                coerced = pd.to_numeric(series, errors="coerce")
                failed = coerced.isna() | (coerced < limit)
                failed_indices = df.index[failed].tolist()

            elif rtype == "max_value":
                limit = float(val) if val is not None else 0
                coerced = pd.to_numeric(series, errors="coerce")
                failed = coerced.isna() | (coerced > limit)
                failed_indices = df.index[failed].tolist()

            elif rtype == "between":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    min_val, max_val = float(val[0]), float(val[1])
                elif isinstance(val, str) and "," in val:
                    parts = val.split(",")
                    min_val, max_val = float(parts[0]), float(parts[1])
                elif isinstance(val, dict):
                    min_val, max_val = float(val.get("min", 0)), float(val.get("max", 100))
                else:
                    min_val, max_val = 0.0, 100.0
                coerced = pd.to_numeric(series, errors="coerce")
                failed = coerced.isna() | (coerced < min_val) | (coerced > max_val)
                failed_indices = df.index[failed].tolist()

            elif rtype == "not_empty":
                missing_mask = series.isna()
                if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                    str_vals = series.astype(str).str.strip().str.lower()
                    sentinel_mask = str_vals.isin(MISSING_SENTINELS) | (str_vals == '')
                    missing_mask = missing_mask | sentinel_mask
                failed_indices = df.index[missing_mask].tolist()

            elif rtype == "unique":
                dup_mask = series.duplicated(keep=False)
                failed_indices = df.index[dup_mask].tolist()

            elif rtype == "contains":
                pattern = str(val) if val is not None else ""
                str_series = series.astype(str)
                failed = ~str_series.str.contains(pattern, case=False, na=False)
                failed_indices = df.index[failed].tolist()

            elif rtype == "not_contains":
                pattern = str(val) if val is not None else ""
                str_series = series.astype(str)
                failed = str_series.str.contains(pattern, case=False, na=False)
                failed_indices = df.index[failed].tolist()

            elif rtype == "regex":
                pattern = str(val) if val is not None else ".*"
                str_series = series.astype(str)
                failed = ~str_series.str.contains(pattern, na=False)
                failed_indices = df.index[failed].tolist()

            elif rtype == "email_format":
                email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                str_series = series.astype(str).str.strip()
                failed = ~str_series.str.match(email_regex, na=False)
                failed_indices = df.index[failed].tolist()

            elif rtype == "phone_format":
                phone_regex = r'^\+?[0-9\s-]{10,14}$'
                str_series = series.astype(str).str.strip()
                failed = ~str_series.str.match(phone_regex, na=False)
                failed_indices = df.index[failed].tolist()

            elif rtype == "positive":
                coerced = pd.to_numeric(series, errors="coerce")
                failed = coerced.isna() | (coerced <= 0)
                failed_indices = df.index[failed].tolist()

            elif rtype == "negative":
                coerced = pd.to_numeric(series, errors="coerce")
                failed = coerced.isna() | (coerced >= 0)
                failed_indices = df.index[failed].tolist()

            else:
                failed_indices = []

            failed_count = len(failed_indices)
            passed_count = total_rows - failed_count
            pass_pct = round((passed_count / max(total_rows, 1)) * 100, 2)

            results.append({
                "rule_name": name,
                "column": col,
                "rule_type": rtype,
                "passed": failed_count == 0,
                "failed_count": failed_count,
                "pass_percentage": pass_pct,
                "failed_row_indices": failed_indices[:200]
            })

        return results

    # ── FEATURE 4: PII Privacy Scanner ──────────────────────────────
    @staticmethod
    def scan_pii(df: pd.DataFrame) -> dict[str, Any]:
        """Scan dataset for sensitive Personally Identifiable Information (PII)."""
        pii_patterns = [
            ("Credit Card", r'\b(?:\d[ -]?){13,16}\b', "high", "Mask credit card numbers before export"),
            ("CNIC Pakistan", r'[0-9]{5}-[0-9]{7}-[0-9]', "high", "Mask CNIC numbers before export"),
            ("National ID", r'\b[A-Z]{1,2}[0-9]{6,9}\b', "high", "Mask National ID values before export"),
            ("Email", r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "medium", "Mask email addresses"),
            ("Phone Pakistan", r'(\+92|0)?[0-9]{10,11}', "medium", "Mask phone numbers"),
            ("Phone International", r'\+?[1-9]\d{1,14}', "medium", "Mask phone numbers"),
            ("IP Address", r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "low", "Mask IP address data"),
            ("URL", r'https?://[^\s]+', "low", "Review URLs for sensitive query parameters"),
        ]

        found_items = []
        high_risk, medium_risk, low_risk = 0, 0, 0

        def mask_sample(val, ptype):
            val_str = str(val).strip()
            if ptype == "Email":
                parts = val_str.split("@")
                if len(parts) == 2:
                    return parts[0][:2] + "***@" + parts[1]
                return val_str[:2] + "***"
            elif ptype in ("Phone Pakistan", "Phone International"):
                return "****-****-" + val_str[-4:] if len(val_str) >= 4 else "****"
            elif ptype == "CNIC Pakistan":
                return val_str[:5] + "-***-*"
            elif ptype == "Credit Card":
                clean = re.sub(r'\D', '', val_str)
                return "**** **** **** " + clean[-4:] if len(clean) >= 4 else "****"
            elif ptype == "IP Address":
                parts = val_str.split(".")
                return f"{parts[0]}.{parts[1]}.***.***" if len(parts) == 4 else val_str
            return val_str[:2] + "***" + val_str[-2:]

        for col in df.columns:
            str_series = df[col].dropna().astype(str)
            if str_series.empty:
                continue

            for ptype, pattern, risk, rec in pii_patterns:
                matches = str_series[str_series.str.contains(pattern, regex=True, na=False)]
                count = len(matches)
                if count > 0:
                    sample_raw = matches.iloc[0]
                    sample_masked = mask_sample(sample_raw, ptype)
                    indices = matches.index[:100].tolist()

                    if risk == "high":
                        high_risk += count
                    elif risk == "medium":
                        medium_risk += count
                    else:
                        low_risk += count

                    found_items.append({
                        "pii_type": ptype,
                        "column_name": col,
                        "count": count,
                        "sample_masked_value": sample_masked,
                        "risk_level": risk,
                        "recommendation": rec,
                        "row_indices": indices
                    })

        return {
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "total_pii_found": len(found_items),
            "pii_items": found_items
        }

    @staticmethod
    def mask_pii(df: pd.DataFrame, column: str, pii_type: str) -> tuple[pd.DataFrame, int]:
        """Mask PII entries in a specified column."""
        df = df.copy()
        if column not in df.columns:
            return df, 0

        count = 0
        def mask_val(v):
            nonlocal count
            if pd.isna(v):
                return v
            v_str = str(v).strip()
            if not v_str or v_str.lower() in MISSING_SENTINELS:
                return v
            count += 1
            if pii_type.lower() in ("email", "email address"):
                parts = v_str.split("@")
                if len(parts) == 2:
                    return parts[0][:2] + "***@" + parts[1]
                return v_str[:2] + "***"
            elif "phone" in pii_type.lower():
                return "****-****-" + v_str[-4:] if len(v_str) >= 4 else "****"
            elif "cnic" in pii_type.lower():
                return v_str[:5] + "-***-*"
            elif "credit" in pii_type.lower() or "card" in pii_type.lower():
                clean = re.sub(r'\D', '', v_str)
                return "**** **** **** " + clean[-4:] if len(clean) >= 4 else "****"
            return v_str[:2] + "***" + v_str[-2:]

        df[column] = df[column].apply(mask_val)
        return df, count

    # ── FEATURE 5: Two File Comparison Engine ────────────────────────
    @staticmethod
    def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Dataset 1", name2: str = "Dataset 2") -> dict[str, Any]:
        """Perform full comparison between two DataFrames."""
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)

        removed_cols = list(cols1 - cols2)
        added_cols = list(cols2 - cols1)
        common_cols = list(cols1 & cols2)

        # Dtype changes
        analyzer1 = DataAnalyzer(df1)
        analyzer2 = DataAnalyzer(df2)
        types1 = analyzer1.data_type_detection()
        types2 = analyzer2.data_type_detection()

        dtype_changes = []
        for c in common_cols:
            if types1.get(c) != types2.get(c):
                dtype_changes.append({
                    "column": c,
                    "type1": types1.get(c),
                    "type2": types2.get(c)
                })

        # Row comparison
        rows1, rows2 = len(df1), len(df2)
        added_rows_count = max(0, rows2 - rows1)
        deleted_rows_count = max(0, rows1 - rows2)
        modified_rows_count = 0
        sample_modified = []

        min_len = min(rows1, rows2)
        for idx in range(min_len):
            row_diffs = []
            for col in common_cols:
                val1, val2 = df1.iloc[idx][col], df2.iloc[idx][col]
                if pd.isna(val1) and pd.isna(val2):
                    continue
                if str(val1) != str(val2):
                    row_diffs.append({
                        "column": col,
                        "old_value": None if pd.isna(val1) else str(val1),
                        "new_value": None if pd.isna(val2) else str(val2)
                    })
            if row_diffs:
                modified_rows_count += 1
                if len(sample_modified) < 20:
                    sample_modified.append({
                        "row_index": idx,
                        "changes": row_diffs
                    })

        # Stats comparison per column
        stats1 = {s["column_name"]: s for s in analyzer1.column_statistics()}
        stats2 = {s["column_name"]: s for s in analyzer2.column_statistics()}

        col_stats_comparison = []
        all_unique_cols = list(dict.fromkeys(list(df1.columns) + list(df2.columns)))
        for c in all_unique_cols:
            st1 = stats1.get(c, {})
            st2 = stats2.get(c, {})
            col_stats_comparison.append({
                "column": c,
                "in_df1": c in cols1,
                "in_df2": c in cols2,
                "missing1": st1.get("missing_count"),
                "missing2": st2.get("missing_count"),
                "mean1": st1.get("mean_value"),
                "mean2": st2.get("mean_value"),
                "unique1": st1.get("unique_count"),
                "unique2": st2.get("unique_count"),
            })

        score1 = analyzer1.quality_score()
        score2 = analyzer2.quality_score()

        return {
            "dataset_1": {"name": name1, "rows": rows1, "columns": len(cols1), "quality_score": score1},
            "dataset_2": {"name": name2, "rows": rows2, "columns": len(cols2), "quality_score": score2},
            "schema_diff": {
                "removed_columns": removed_cols,
                "added_columns": added_cols,
                "common_columns": common_cols,
                "dtype_changes": dtype_changes
            },
            "row_diff": {
                "added_rows_count": added_rows_count,
                "deleted_rows_count": deleted_rows_count,
                "modified_rows_count": modified_rows_count,
                "sample_modified_rows": sample_modified
            },
            "column_stats_comparison": col_stats_comparison
        }

    # ── FEATURE 6: Column Relationship Detection Engine ──────────────
    @staticmethod
    def analyze_relationships(df: pd.DataFrame) -> dict[str, Any]:
        """Analyze numerical correlation matrix, duplicate columns, and low variance features."""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = {}
        high_correlations = []

        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_df = numeric_df.corr().round(3)
            # Replace NaNs with 0
            corr_df = corr_df.fillna(0)
            
            for col in corr_df.columns:
                correlation_matrix[col] = corr_df[col].to_dict()

            cols = list(corr_df.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = float(corr_df.iloc[i, j])
                    abs_val = abs(val)
                    c1, c2 = cols[i], cols[j]
                    if abs_val > 0.85:
                        rec = "Nearly identical — consider removing one" if abs_val > 0.95 else "Highly correlated feature pair"
                        high_correlations.append({
                            "col1": c1,
                            "col2": c2,
                            "correlation": val,
                            "abs_correlation": abs_val,
                            "recommendation": rec
                        })

        # Duplicate column values
        duplicate_col_pairs = []
        cols = list(df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                if df[c1].equals(df[c2]):
                    duplicate_col_pairs.append({
                        "col1": c1,
                        "col2": c2,
                        "recommendation": f"Column '{c2}' is an exact duplicate of '{c1}' — remove one."
                    })

        # Low variance / ID columns / Constant
        id_columns = []
        low_variance_columns = []
        constant_cols = []

        for col in df.columns:
            series = df[col].dropna()
            total = len(df)
            if series.empty:
                continue
            unique_count = series.nunique()
            if unique_count == 1:
                constant_cols.append(col)
            elif unique_count == total and (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
                id_columns.append({
                    "column": col,
                    "recommendation": "This looks like an ID column — consider excluding from numerical analysis."
                })
            elif pd.api.types.is_numeric_dtype(series):
                std_val = series.std()
                mean_val = series.mean()
                if pd.notna(std_val) and pd.notna(mean_val) and abs(mean_val) > 0:
                    if std_val < 0.01 * abs(mean_val):
                        low_variance_columns.append({
                            "column": col,
                            "std": round(float(std_val), 4),
                            "mean": round(float(mean_val), 4),
                            "recommendation": f"Column '{col}' has near-zero variance (std={round(float(std_val),4)})."
                        })

        return {
            "correlation_matrix": correlation_matrix,
            "high_correlations": high_correlations,
            "duplicate_column_pairs": duplicate_col_pairs,
            "id_columns": id_columns,
            "low_variance_columns": low_variance_columns,
            "constant_columns": constant_cols
        }
