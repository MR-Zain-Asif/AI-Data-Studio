"""Data cleaning engine — all fixes using pure pandas and scikit-learn."""

import pandas as pd
import numpy as np
import re
import difflib
import unicodedata
from typing import Optional, Any, List, Dict, Tuple, Union
from dateutil import parser as date_parser

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder
)
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except Exception:
    IterativeImputer = None

# Global memory storage for user saved pipelines
SAVED_PIPELINES: dict[str, list[dict[str, Any]]] = {}

# Preset Pipelines definition
PRESET_PIPELINES: dict[str, dict[str, Any]] = {
    "Sales Data": {
        "description": "Trims whitespace, removes duplicates, smart-fills missing values, caps outliers",
        "icon": "ti-shopping-cart",
        "steps": [
            {"operation": "trim_whitespace"},
            {"operation": "remove_duplicates"},
            {"operation": "fill_all_missing"},
            {"operation": "fix_outliers_all", "method": "cap"}
        ]
    },
    "HR Data": {
        "description": "Trims whitespace, removes duplicates, fills mode, standardizes text to Title Case",
        "icon": "ti-users",
        "steps": [
            {"operation": "trim_whitespace"},
            {"operation": "remove_duplicates"},
            {"operation": "fill_all_missing"},
            {"operation": "standardize_case_all", "case": "title"}
        ]
    },
    "Healthcare Data": {
        "description": "Trims whitespace, removes duplicates, fills median values, removes outliers",
        "icon": "ti-activity",
        "steps": [
            {"operation": "trim_whitespace"},
            {"operation": "remove_duplicates"},
            {"operation": "fill_all_missing"},
            {"operation": "fix_outliers_all", "method": "remove"}
        ]
    },
    "Finance Data": {
        "description": "Trims whitespace, removes duplicates, fills missing mean, caps outliers, drops constant columns",
        "icon": "ti-currency-dollar",
        "steps": [
            {"operation": "trim_whitespace"},
            {"operation": "remove_duplicates"},
            {"operation": "fill_all_missing"},
            {"operation": "fix_outliers_all", "method": "cap"},
            {"operation": "remove_constant_columns"}
        ]
    }
}


class DataCleaner:
    """Performs data cleaning operations on a pandas DataFrame."""

    @staticmethod
    def _get_missing_mask(series: pd.Series) -> pd.Series:
        """Return boolean mask where True indicates a missing value (native NaN or sentinel string)."""
        native_missing = series.isna()
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            from services.analyzer import MISSING_SENTINELS
            str_vals = series.astype(str).str.strip()
            sentinel_mask = str_vals.str.lower().isin(MISSING_SENTINELS)
            whitespace_only_mask = (str_vals == '') | series.astype(str).str.fullmatch(r'\s+')
            return native_missing | sentinel_mask | whitespace_only_mask
        return native_missing

    @staticmethod
    def fill_missing(df: pd.DataFrame, column: str, method: str, value: Optional[str] = None) -> tuple[pd.DataFrame, int]:
        """Fill missing values in a column using the specified method."""
        df = df.copy()
        if column not in df.columns:
            return df, 0

        missing_mask = DataCleaner._get_missing_mask(df[column])
        rows_affected = int(missing_mask.sum())

        if rows_affected == 0:
            return df, 0

        # Convert missing sentinels to NaN so pandas fillna operates on them
        df.loc[missing_mask, column] = np.nan

        if method == "mean":
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            else:
                mode_vals = df[column].mode()
                if not mode_vals.empty:
                    df[column] = df[column].fillna(mode_vals.iloc[0])
        elif method == "median":
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            else:
                mode_vals = df[column].mode()
                if not mode_vals.empty:
                    df[column] = df[column].fillna(mode_vals.iloc[0])
        elif method == "mode":
            mode_vals = df[column].mode()
            if not mode_vals.empty:
                df[column] = df[column].fillna(mode_vals.iloc[0])
        elif method == "custom_value":
            if value is not None:
                try:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        fill_val = float(value)
                    else:
                        fill_val = value
                except (ValueError, TypeError):
                    fill_val = value
                df[column] = df[column].fillna(fill_val)
        else:
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            else:
                mode_vals = df[column].mode()
                if not mode_vals.empty:
                    df[column] = df[column].fillna(mode_vals.iloc[0])

        return df, rows_affected

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove exact duplicate rows. Returns (new_df, rows_removed)."""
        original_len = len(df)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        rows_removed = original_len - len(df)
        return df, rows_removed

    @staticmethod
    def trim_whitespace(df: pd.DataFrame, column: Optional[str] = None) -> tuple[pd.DataFrame, int]:
        """Strip leading/trailing spaces. If column is None, apply to all text columns."""
        df = df.copy()
        total_affected = 0

        cols = [column] if column else [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]

        for col in cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                original = df[col].copy()
                df[col] = df[col].astype(str).str.strip()
                df.loc[original.isna(), col] = np.nan
                changed = (original.fillna("__NULL__") != df[col].fillna("__NULL__")).sum()
                total_affected += int(changed)

        return df, total_affected

    @staticmethod
    def standardize_case(df: pd.DataFrame, column: Optional[str] = None, case: str = "lower") -> tuple[pd.DataFrame, int]:
        """Standardize text case: upper, lower, or title. If column is None, apply to all text columns."""
        df = df.copy()
        cols = [column] if column else [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
        total_changed = 0

        for col in cols:
            if col not in df.columns:
                continue
            original = df[col].copy()
            not_null = df[col].notna()

            if case == "upper":
                df.loc[not_null, col] = df.loc[not_null, col].astype(str).str.upper()
            elif case == "lower":
                df.loc[not_null, col] = df.loc[not_null, col].astype(str).str.lower()
            elif case == "title":
                df.loc[not_null, col] = df.loc[not_null, col].astype(str).str.title()

            changed = (original.fillna("__NULL__") != df[col].fillna("__NULL__")).sum()
            total_changed += int(changed)

        return df, total_changed

    @staticmethod
    def remove_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove columns with 80%+ missing values."""
        df = df.copy()
        cols_to_drop = []
        for col in df.columns:
            missing_mask = DataCleaner._get_missing_mask(df[col])
            missing_pct = missing_mask.sum() / len(df) * 100 if len(df) > 0 else 0
            if missing_pct >= 80:
                cols_to_drop.append(col)
        df = df.drop(columns=cols_to_drop)
        return df, len(cols_to_drop)

    @staticmethod
    def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove columns with a single unique value."""
        df = df.copy()
        cols_to_drop = []
        for col in df.columns:
            non_null = df[col].dropna()
            if not non_null.empty and non_null.nunique() == 1:
                cols_to_drop.append(col)
        df = df.drop(columns=cols_to_drop)
        return df, len(cols_to_drop)

    @staticmethod
    def fix_outliers(df: pd.DataFrame, column: str, method: str = "cap") -> tuple[pd.DataFrame, int]:
        """Fix outliers using IQR method: cap, impute, median, mean, or remove."""
        df = df.copy()
        if column not in df.columns:
            return df, 0

        if pd.api.types.is_bool_dtype(df[column]):
            return df, 0

        coerced = pd.to_numeric(df[column], errors="coerce").astype(float)
        series = coerced.dropna()
        if series.empty:
            return df, 0

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            tol = 1e-9 * iqr
            cap_lower = lower + 1e-5 * iqr
            cap_upper = upper - 1e-5 * iqr
        else:
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad > 0:
                lower = median - 3.0 * 1.4826 * mad
                upper = median + 3.0 * 1.4826 * mad
                tol = 1e-9 * mad
                cap_lower = lower + 1e-5 * mad
                cap_upper = upper - 1e-5 * mad
            else:
                if series.nunique() > 1:
                    lower = median
                    upper = median
                    tol = 1e-9
                    cap_lower = lower + 1e-7
                    cap_upper = upper - 1e-7
                else:
                    return df, 0

        outlier_mask = (coerced < lower - tol) | (coerced > upper + tol)
        outlier_mask = outlier_mask & coerced.notna()
        rows_affected = int(outlier_mask.sum())

        if rows_affected == 0:
            return df, 0

        if method in ("cap", "impute", "median", "mean"):
            if not pd.api.types.is_numeric_dtype(df[column]):
                df[column] = pd.to_numeric(df[column], errors="coerce")

            if pd.api.types.is_integer_dtype(df[column]):
                df[column] = df[column].astype(float)

        if method == "cap":
            df.loc[coerced < lower - tol, column] = cap_lower
            df.loc[coerced > upper + tol, column] = cap_upper
        elif method in ("impute", "median"):
            median_val = float(series.median())
            df.loc[outlier_mask, column] = median_val
        elif method == "mean":
            mean_val = float(series.mean())
            df.loc[outlier_mask, column] = mean_val
        elif method == "remove":
            df = df[~outlier_mask].reset_index(drop=True)

        return df, rows_affected

    @staticmethod
    def convert_dates(df: pd.DataFrame, column: str, fmt: str = "%Y-%m-%d") -> tuple[pd.DataFrame, int]:
        """Standardize date format in a column."""
        df = df.copy()
        if column not in df.columns:
            return df, 0

        converted = 0
        new_values = []
        missing_mask = DataCleaner._get_missing_mask(df[column])

        for idx in df.index:
            val = df.loc[idx, column]
            if missing_mask.loc[idx]:
                new_values.append(np.nan)
                continue
            try:
                parsed = date_parser.parse(str(val), fuzzy=True)
                new_values.append(parsed.strftime(fmt))
                converted += 1
            except (ValueError, OverflowError, TypeError):
                new_values.append(val)

        df[column] = new_values
        return df, converted

    @staticmethod
    def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> tuple[pd.DataFrame, int]:
        """Rename a column."""
        df = df.copy()
        if old_name not in df.columns:
            return df, 0
        df = df.rename(columns={old_name: new_name})
        return df, 1

    @staticmethod
    def delete_column(df: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame, int]:
        """Drop a column."""
        df = df.copy()
        if column_name not in df.columns:
            return df, 0
        df = df.drop(columns=[column_name])
        return df, 1

    @staticmethod
    def change_dtype(df: pd.DataFrame, column: str, dtype: str) -> tuple[pd.DataFrame, int]:
        """Convert column to a different data type."""
        df = df.copy()
        if column not in df.columns:
            return df, 0

        rows_affected = int(df[column].notna().sum())
        try:
            if dtype == "int":
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
            elif dtype == "str":
                df[column] = df[column].astype(str)
                df.loc[df[column] == "nan", column] = np.nan
            elif dtype == "datetime":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            else:
                return df, 0
        except Exception:
            return df, 0

        return df, rows_affected

    @staticmethod
    def fill_all_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Smart fill: mean for numbers, mode for text."""
        df = df.copy()
        total_filled = 0

        for col in df.columns:
            missing_mask = DataCleaner._get_missing_mask(df[col])
            missing = int(missing_mask.sum())
            if missing == 0:
                continue
            df.loc[missing_mask, col] = np.nan
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col] = df[col].fillna(mean_val)
                    total_filled += missing
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                    total_filled += missing

        return df, total_filled

    @staticmethod
    def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[dict]]:
        """Run all safe fixes automatically in order."""
        steps: list[dict] = []
        total_affected = 0

        # 1: Remove empty columns
        df, n = DataCleaner.remove_empty_columns(df)
        steps.append({"step": 1, "name": "Removing empty columns", "status": "done", "affected": n})
        total_affected += n

        # 2: Remove constant columns
        df, n = DataCleaner.remove_constant_columns(df)
        steps.append({"step": 2, "name": "Removing constant columns", "status": "done", "affected": n})
        total_affected += n

        # 3: Remove duplicate rows
        df, n = DataCleaner.remove_duplicates(df)
        steps.append({"step": 3, "name": "Removing duplicate rows", "status": "done", "affected": n})
        total_affected += n

        # 4: Trim whitespace
        df, n = DataCleaner.trim_whitespace(df)
        steps.append({"step": 4, "name": "Trimming whitespace", "status": "done", "affected": n})
        total_affected += n

        # 5: Smart fill missing
        df, n = DataCleaner.fill_all_missing(df)
        steps.append({"step": 5, "name": "Filling missing values", "status": "done", "affected": n})
        total_affected += n

        # 6: Fix outliers
        from services.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)
        dtypes = analyzer.data_type_detection()
        outlier_total = 0
        for col in df.columns:
            if dtypes.get(col) == "number":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df, n = DataCleaner.fix_outliers(df, col, "cap")
                outlier_total += n
        steps.append({"step": 6, "name": "Fixing outliers", "status": "done", "affected": outlier_total})
        total_affected += outlier_total

        # 7: Standardize dates
        analyzer = DataAnalyzer(df)
        dtypes = analyzer.data_type_detection()
        date_total = 0
        for col, dtype in dtypes.items():
            if dtype == "date" and col in df.columns:
                df, n = DataCleaner.convert_dates(df, col, "%Y-%m-%d")
                date_total += n
        steps.append({"step": 7, "name": "Standardizing date formats", "status": "done", "affected": date_total})
        total_affected += date_total

        return df, total_affected, steps

    # ── FEATURE 1: Pipeline Functions ───────────────────────────────
    @staticmethod
    def save_pipeline(name: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Save a new pipeline in memory."""
        SAVED_PIPELINES[name] = steps
        return {"name": name, "step_count": len(steps)}

    @staticmethod
    def list_pipelines() -> dict[str, Any]:
        """Return all saved and preset pipelines."""
        user_pipelines = {k: {"steps_count": len(v), "steps": v, "is_preset": False} for k, v in SAVED_PIPELINES.items()}
        presets = {k: {"steps_count": len(v["steps"]), "description": v["description"], "icon": v["icon"], "steps": v["steps"], "is_preset": True} for k, v in PRESET_PIPELINES.items()}
        return {"presets": presets, "user_pipelines": user_pipelines}

    # ── FEATURE 9: Smart Duplicate Column Merger ────────────────────
    @staticmethod
    def find_similar_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Find pairs of columns that have similar names or high value overlap."""
        cols = list(df.columns)
        similar_pairs = []

        def norm_name(s):
            return re.sub(r'[^a-z0-9]', '', str(s).lower())

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                n1, n2 = norm_name(col1), norm_name(col2)
                
                # Check name similarity
                name_sim = difflib.SequenceMatcher(None, n1, n2).ratio()
                
                # Check value similarity if non-empty
                s1 = df[col1].dropna().astype(str).str.strip().str.lower()
                s2 = df[col2].dropna().astype(str).str.strip().str.lower()
                val_sim = 0.0
                
                if len(s1) > 0 and len(s2) > 0:
                    common_len = min(len(s1), len(s2))
                    match_count = (s1.iloc[:common_len].values == s2.iloc[:common_len].values).sum()
                    val_sim = float(match_count / common_len)

                score = max(name_sim, val_sim)
                if score >= 0.75:
                    similar_pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "similarity_score": round(score * 100, 1),
                        "name_similarity": round(name_sim * 100, 1),
                        "value_overlap": round(val_sim * 100, 1),
                        "sample_col1": df[col1].dropna().head(3).tolist(),
                        "sample_col2": df[col2].dropna().head(3).tolist(),
                        "recommendation": f"Columns '{col1}' and '{col2}' are {round(score*100)}% similar. Consider merging them."
                    })

        return similar_pairs

    @staticmethod
    def merge_columns(df: pd.DataFrame, col1: str, col2: str, strategy: str, new_name: str) -> tuple[pd.DataFrame, int]:
        """Merge two columns using chosen strategy and delete originals."""
        df = df.copy()
        if col1 not in df.columns or col2 not in df.columns:
            return df, 0

        rows = len(df)
        s1 = df[col1]
        s2 = df[col2]
        merged_series = pd.Series(index=df.index, dtype=object)

        if strategy == "prefer_first":
            merged_series = s1.combine_first(s2)
        elif strategy == "prefer_second":
            merged_series = s2.combine_first(s1)
        elif strategy == "concatenate":
            part1 = s1.fillna('').astype(str)
            part2 = s2.fillna('').astype(str)
            merged_series = (part1 + " " + part2).str.strip()
            merged_series = merged_series.replace('', np.nan)
        elif strategy == "sum":
            num1 = pd.to_numeric(s1, errors="coerce").fillna(0)
            num2 = pd.to_numeric(s2, errors="coerce").fillna(0)
            merged_series = num1 + num2
        elif strategy == "average":
            num1 = pd.to_numeric(s1, errors="coerce")
            num2 = pd.to_numeric(s2, errors="coerce")
            merged_series = pd.concat([num1, num2], axis=1).mean(axis=1)
        else:
            merged_series = s1.combine_first(s2)

        # Drop original columns and insert merged column
        df = df.drop(columns=[col1, col2])
        df[new_name] = merged_series
        return df, rows

    @staticmethod
    def knn_impute(df: pd.DataFrame, column: str, n_neighbors: int = 5) -> Tuple[pd.DataFrame, int]:
        """Fill missing values in target numeric column using KNNImputer with other numeric columns."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if column not in num_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        missing_mask = df[column].isna()
        rows_affected = int(missing_mask.sum())
        if rows_affected == 0:
            return df, 0

        # Run KNNImputer on numeric subset
        imputer = KNNImputer(n_neighbors=max(1, min(n_neighbors, max(len(df) - 1, 1))))
        imputed_vals = imputer.fit_transform(df[num_cols])
        col_idx = num_cols.index(column)
        df[column] = imputed_vals[:, col_idx]
        return df, rows_affected

    @staticmethod
    def regression_impute(df: pd.DataFrame, target_col: str, predictor_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """Train LinearRegression on complete rows and predict missing values in target_col."""
        df = df.copy()
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame")

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        missing_mask = df[target_col].isna()
        rows_affected = int(missing_mask.sum())
        if rows_affected == 0:
            return df, 0

        if not predictor_cols:
            predictor_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        if not predictor_cols:
            # Fallback to median
            df[target_col] = df[target_col].fillna(df[target_col].median())
            return df, rows_affected

        # Prepare train and predict sets
        valid_rows = df[~missing_mask].dropna(subset=predictor_cols)
        if len(valid_rows) < 3:
            df[target_col] = df[target_col].fillna(df[target_col].median())
            return df, rows_affected

        X_train = valid_rows[predictor_cols].values
        y_train = valid_rows[target_col].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Predict missing rows
        missing_rows = df[missing_mask].copy()
        # Impute predictors if missing
        X_pred = missing_rows[predictor_cols].fillna(df[predictor_cols].mean()).values
        preds = lr.predict(X_pred)
        df.loc[missing_mask, target_col] = preds
        return df, rows_affected

    @staticmethod
    def iterative_impute(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Use IterativeImputer (MICE method) across numeric features."""
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return df, 0

        initial_missing = int(df[num_cols].isna().sum().sum())
        if initial_missing == 0:
            return df, 0

        if IterativeImputer is not None:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            imputed_vals = imputer.fit_transform(df[num_cols])
            df[num_cols] = imputed_vals
        else:
            knn = KNNImputer(n_neighbors=5)
            df[num_cols] = knn.fit_transform(df[num_cols])

        return df, initial_missing

    @staticmethod
    def isolation_forest_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, contamination: float = 0.05) -> Dict[str, Any]:
        """Detect multi-feature outliers using IsolationForest."""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = [c for c in columns if c in df.columns]

        if not columns:
            return {"outlier_indices": [], "outlier_count": 0, "per_column_stats": {}}

        clean_data = df[columns].fillna(df[columns].median())
        iso = IsolationForest(contamination=max(0.01, min(0.4, contamination)), random_state=42)
        preds = iso.fit_predict(clean_data)
        outlier_mask = preds == -1
        outlier_indices = df.index[outlier_mask].tolist()

        per_col_stats = {}
        for col in columns:
            col_outliers = df.loc[outlier_mask, col].dropna()
            per_col_stats[col] = {
                "outlier_count": len(col_outliers),
                "min": round(float(col_outliers.min()), 2) if len(col_outliers) else None,
                "max": round(float(col_outliers.max()), 2) if len(col_outliers) else None
            }

        return {
            "outlier_indices": outlier_indices,
            "outlier_count": len(outlier_indices),
            "per_column_stats": per_col_stats
        }

    @staticmethod
    def lof_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect local density-based outliers using LocalOutlierFactor."""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = [c for c in columns if c in df.columns]

        if not columns or len(df) < 5:
            return {"outlier_indices": [], "outlier_count": 0, "per_column_stats": {}}

        clean_data = df[columns].fillna(df[columns].median())
        n_neighbors = min(20, max(2, len(clean_data) - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        preds = lof.fit_predict(clean_data)
        outlier_mask = preds == -1
        outlier_indices = df.index[outlier_mask].tolist()

        per_col_stats = {}
        for col in columns:
            col_outliers = df.loc[outlier_mask, col].dropna()
            per_col_stats[col] = {
                "outlier_count": len(col_outliers),
                "min": round(float(col_outliers.min()), 2) if len(col_outliers) else None,
                "max": round(float(col_outliers.max()), 2) if len(col_outliers) else None
            }

        return {
            "outlier_indices": outlier_indices,
            "outlier_count": len(outlier_indices),
            "per_column_stats": per_col_stats
        }

    @staticmethod
    def winsorize_column(df: pd.DataFrame, column: str, limits: Tuple[float, float] = (0.05, 0.05)) -> Tuple[pd.DataFrame, int]:
        """Cap extreme values at given lower and upper percentile limits."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        s = pd.to_numeric(df[column], errors="coerce")
        clean = s.dropna()
        if len(clean) < 5:
            return df, 0

        low_p = float(limits[0])
        high_p = 1.0 - float(limits[1])
        q_low = float(clean.quantile(low_p))
        q_high = float(clean.quantile(high_p))

        mask = (s < q_low) | (s > q_high)
        count = int(mask.sum())
        df[column] = s.clip(lower=q_low, upper=q_high)
        return df, count

    @staticmethod
    def clean_text_column(df: pd.DataFrame, column: str, operations: List[str]) -> Tuple[pd.DataFrame, int]:
        """Apply a sequence of text normalization operations on a column."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        s = df[column].astype(str)
        orig = s.copy()

        for op in operations:
            op_l = op.lower().strip()
            if op_l == "strip":
                s = s.str.strip()
            elif op_l == "lower":
                s = s.str.lower()
            elif op_l == "upper":
                s = s.str.upper()
            elif op_l == "title":
                s = s.str.title()
            elif op_l == "remove_special":
                s = s.apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
            elif op_l == "remove_numbers":
                s = s.apply(lambda x: re.sub(r'\d+', '', str(x)))
            elif op_l == "remove_extra_spaces":
                s = s.apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
            elif op_l == "normalize_unicode":
                s = s.apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8'))

        df[column] = s
        affected = int((orig != s).sum())
        return df, affected

    @staticmethod
    def standardize_categories(df: pd.DataFrame, column: str, mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Any]:
        """Standardize categorical text values using mapping or auto-generate fuzzy matches."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        s = df[column].dropna().astype(str)
        if mapping:
            df[column] = df[column].replace(mapping)
            affected = int(s.isin(mapping.keys()).sum())
            return df, affected
        else:
            # Auto detect similar categories via fuzzy matching
            uniques = s.unique().tolist()
            suggestions: Dict[str, str] = {}
            for i in range(len(uniques)):
                for j in range(i + 1, len(uniques)):
                    u1, u2 = uniques[i], uniques[j]
                    sim = difflib.SequenceMatcher(None, u1.lower().strip(), u2.lower().strip()).ratio()
                    if sim >= 0.82 and u1.lower().strip() != u2.lower().strip():
                        # Pick title case or longer as canonical
                        canonical = u1.title() if len(u1) >= len(u2) else u2.title()
                        suggestions[u1] = canonical
                        suggestions[u2] = canonical
            return df, suggestions

    @staticmethod
    def parse_and_standardize_dates(df: pd.DataFrame, column: str, target_format: str = "%Y-%m-%d") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse mixed date formats into target string format."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        s = df[column]
        converted = 0
        failed = 0
        failed_samples = []
        new_vals = []

        for val in s:
            if pd.isna(val) or str(val).strip() == "":
                new_vals.append(None)
                continue
            try:
                dt = date_parser.parse(str(val))
                new_vals.append(dt.strftime(target_format))
                converted += 1
            except Exception:
                new_vals.append(val)
                failed += 1
                if len(failed_samples) < 5:
                    failed_samples.append(str(val))

        df[column] = new_vals
        report = {
            "converted_count": converted,
            "failed_count": failed,
            "failed_samples": failed_samples
        }
        return df, report

    @staticmethod
    def extract_date_features(df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, List[str]]:
        """Extract rich date components: year, month, day, dayofweek, quarter, week_of_year, etc."""
        df = df.copy()
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not in DataFrame")

        dates = pd.to_datetime(df[date_column], errors="coerce")
        col_prefix = date_column

        new_cols = []
        df[f"{col_prefix}_year"] = dates.dt.year
        new_cols.append(f"{col_prefix}_year")

        df[f"{col_prefix}_month"] = dates.dt.month
        new_cols.append(f"{col_prefix}_month")

        df[f"{col_prefix}_day"] = dates.dt.day
        new_cols.append(f"{col_prefix}_day")

        df[f"{col_prefix}_dayofweek"] = dates.dt.dayofweek
        new_cols.append(f"{col_prefix}_dayofweek")

        df[f"{col_prefix}_quarter"] = dates.dt.quarter
        new_cols.append(f"{col_prefix}_quarter")

        df[f"{col_prefix}_week_of_year"] = dates.dt.isocalendar().week.astype(float)
        new_cols.append(f"{col_prefix}_week_of_year")

        df[f"{col_prefix}_is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        new_cols.append(f"{col_prefix}_is_weekend")

        df[f"{col_prefix}_is_month_start"] = dates.dt.is_month_start.astype(int)
        new_cols.append(f"{col_prefix}_is_month_start")

        df[f"{col_prefix}_is_month_end"] = dates.dt.is_month_end.astype(int)
        new_cols.append(f"{col_prefix}_is_month_end")

        return df, new_cols

    @staticmethod
    def scale_features(df: pd.DataFrame, columns: List[str], method: str) -> Tuple[pd.DataFrame, Any]:
        """Scale continuous numeric features (minmax, standard, robust, log, sqrt)."""
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df, None

        method_l = method.lower()
        scaler = None

        if method_l == "minmax":
            scaler = MinMaxScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols].fillna(df[valid_cols].median()))
        elif method_l == "standard":
            scaler = StandardScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols].fillna(df[valid_cols].median()))
        elif method_l == "robust":
            scaler = RobustScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols].fillna(df[valid_cols].median()))
        elif method_l == "log":
            for c in valid_cols:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                df[c] = np.log1p(np.maximum(s, 0))
            scaler = "log"
        elif method_l == "sqrt":
            for c in valid_cols:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                df[c] = np.sqrt(np.maximum(s, 0))
            scaler = "sqrt"
        else:
            raise ValueError(f"Unknown scaling method '{method}'")

        return df, scaler

    @staticmethod
    def inverse_scale(df: pd.DataFrame, columns: List[str], scaler: Any) -> pd.DataFrame:
        """Reverse feature scaling back to original domain representation."""
        df = df.copy()
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols or scaler is None:
            return df

        if hasattr(scaler, "inverse_transform"):
            df[valid_cols] = scaler.inverse_transform(df[valid_cols])
        elif scaler == "log":
            for c in valid_cols:
                df[c] = np.expm1(df[c])
        elif scaler == "sqrt":
            for c in valid_cols:
                df[c] = df[c] ** 2
        return df

    @staticmethod
    def encode_categorical(df: pd.DataFrame, column: str, method: str, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Any]:
        """Encode categorical column into numerical format."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        method_l = method.lower()
        encoder = None

        if method_l == "onehot":
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True, dtype=int)
            df = df.drop(columns=[column]).join(dummies)
            encoder = {"method": "onehot", "columns": dummies.columns.tolist()}
        elif method_l == "label":
            le = LabelEncoder()
            s = df[column].fillna("Missing").astype(str)
            df[column] = le.fit_transform(s)
            encoder = le
        elif method_l == "ordinal":
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            s = df[[column]].fillna("Missing").astype(str)
            df[column] = oe.fit_transform(s)
            encoder = oe
        elif method_l == "frequency":
            freq = df[column].value_counts(normalize=True).to_dict()
            df[column] = df[column].map(freq).fillna(0)
            encoder = {"method": "frequency", "map": freq}
        elif method_l == "binary":
            le = LabelEncoder()
            nums = le.fit_transform(df[column].fillna("Missing").astype(str))
            max_val = max(nums) if len(nums) else 0
            n_bits = max(1, int(math.ceil(math.log2(max_val + 1)))) if max_val > 0 else 1
            for b in range(n_bits):
                df[f"{column}_bin_{b}"] = [(x >> b) & 1 for x in nums]
            df = df.drop(columns=[column])
            encoder = {"method": "binary", "bits": n_bits}
        elif method_l == "target" and target_col and target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors="coerce")
            mean_map = target_series.groupby(df[column]).mean().to_dict()
            df[column] = df[column].map(mean_map).fillna(target_series.mean())
            encoder = {"method": "target", "map": mean_map}
        else:
            # Default to label encoding
            le = LabelEncoder()
            s = df[column].fillna("Missing").astype(str)
            df[column] = le.fit_transform(s)
            encoder = le

        return df, encoder

    @staticmethod
    def split_column(df: pd.DataFrame, column: str, delimiter: str, new_col_names: List[str]) -> pd.DataFrame:
        """Split a single column by delimiter into multiple columns."""
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")

        split_df = df[column].astype(str).str.split(delimiter, n=len(new_col_names) - 1, expand=True)
        for i, col_name in enumerate(new_col_names):
            if i < split_df.shape[1]:
                df[col_name] = split_df[i].str.strip()
            else:
                df[col_name] = ""
        return df

    @staticmethod
    def merge_multiple_columns(df: pd.DataFrame, columns: List[str], separator: str, new_col_name: str) -> pd.DataFrame:
        """Combine multiple columns into one with custom separator."""
        df = df.copy()
        valid = [c for c in columns if c in df.columns]
        if not valid:
            raise ValueError("No valid columns to merge")

        df[new_col_name] = df[valid[0]].fillna('').astype(str)
        for c in valid[1:]:
            df[new_col_name] = df[new_col_name] + separator + df[c].fillna('').astype(str)
        return df

    @staticmethod
    def create_calculated_column(df: pd.DataFrame, new_col_name: str, formula: str) -> pd.DataFrame:
        """Evaluate a safe mathematical expression across DataFrame columns."""
        df = df.copy()
        # Clean formula and safely evaluate
        cleaned_expr = formula.strip()
        df[new_col_name] = df.eval(cleaned_expr)
        return df
