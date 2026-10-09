"""DataStudio AI — All data science tools fully implemented.

Each tool returns: (new_df_or_result, message, rows_affected, code_string)
For analysis tools: returns (result_dict, message, 0, code_string)
"""

import json
import time
import numpy as np
import pandas as pd


class DataScienceTools:
    """Complete library of 20+ data science tools."""

    # ── QUALITY SCORE ───────────────────────────────────────────────────────

    def _calculate_quality_score(self, df: pd.DataFrame, issues: list) -> float:
        total_cells = df.size
        if total_cells == 0:
            return 100.0
        missing_pct = df.isnull().sum().sum() / total_cells
        dup_pct = df.duplicated().sum() / max(len(df), 1)
        score = 100 - (missing_pct * 40) - (dup_pct * 20) - min(len(issues) * 2, 20)
        return max(0.0, min(100.0, round(score, 1)))

    # ── ANALYSIS TOOLS ──────────────────────────────────────────────────────

    def analyze_quality(self, df: pd.DataFrame) -> dict:
        """Detect all data quality issues: missing, duplicates, outliers, whitespace."""
        issues = []

        # Missing values
        missing = df.isnull().sum()
        for col, count in missing[missing > 0].items():
            pct = round(count / len(df) * 100, 1)
            severity = "critical" if pct > 50 else "warning" if pct > 20 else "info"
            issues.append({
                "type": "missing_values", "column": col,
                "count": int(count), "percentage": pct,
                "severity": severity, "fix": "fill_missing"
            })

        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = round(dup_count / len(df) * 100, 1)
            issues.append({
                "type": "duplicate_rows", "column": None,
                "count": int(dup_count), "percentage": pct,
                "severity": "critical" if pct > 10 else "warning",
                "fix": "remove_duplicates"
            })

        # Outliers (numeric)
        for col in df.select_dtypes(include='number').columns:
            clean = df[col].dropna()
            if len(clean) == 0:
                continue
            Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
            IQR = Q3 - Q1
            outliers = int(((clean < Q1 - 1.5 * IQR) | (clean > Q3 + 1.5 * IQR)).sum())
            if outliers > 0:
                issues.append({
                    "type": "outliers", "column": col,
                    "count": outliers,
                    "percentage": round(outliers / len(df) * 100, 1),
                    "severity": "warning", "fix": "fix_outliers"
                })

        # Whitespace
        for col in df.select_dtypes(include='object').columns:
            try:
                ws_count = int(df[col].dropna().astype(str).str.strip().ne(
                    df[col].dropna().astype(str)
                ).sum())
                if ws_count > 0:
                    issues.append({
                        "type": "whitespace", "column": col,
                        "count": ws_count, "percentage": round(ws_count / len(df) * 100, 1),
                        "severity": "info", "fix": "standardize_text"
                    })
            except Exception:
                pass

        score = self._calculate_quality_score(df, issues)
        return {
            "issues": issues,
            "quality_score": score,
            "total_issues": len(issues),
            "shape": list(df.shape),
            "missing_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        }

    def profile_columns(self, df: pd.DataFrame) -> dict:
        """Deep per-column profile with stats for each dtype."""
        profiles = {}
        for col in df.columns:
            profile = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().sum() / max(len(df), 1) * 100, 1),
                "unique": int(df[col].nunique()),
                "unique_pct": round(df[col].nunique() / max(len(df), 1) * 100, 1)
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                clean = df[col].dropna()
                profile.update({
                    "type": "numeric",
                    "mean": round(float(clean.mean()), 4) if len(clean) else None,
                    "median": round(float(clean.median()), 4) if len(clean) else None,
                    "std": round(float(clean.std()), 4) if len(clean) else None,
                    "min": round(float(clean.min()), 4) if len(clean) else None,
                    "max": round(float(clean.max()), 4) if len(clean) else None,
                    "skewness": round(float(clean.skew()), 4) if len(clean) > 2 else None,
                    "kurtosis": round(float(clean.kurt()), 4) if len(clean) > 3 else None,
                })
            else:
                top = df[col].value_counts().head(5).to_dict()
                profile.update({
                    "type": "categorical",
                    "top_values": {str(k): int(v) for k, v in top.items()},
                    "avg_length": round(
                        df[col].dropna().astype(str).str.len().mean(), 1
                    ) if df[col].notna().any() else 0
                })
            profiles[col] = profile
        return profiles

    def find_correlations(self, df: pd.DataFrame) -> dict:
        """Compute correlation matrix and highlight strong relationships."""
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            return {"matrix": {}, "high_correlations": []}

        corr = numeric_df.corr().round(3)
        high_corr = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    high_corr.append({
                        "col1": cols[i], "col2": cols[j],
                        "correlation": round(float(val), 3),
                        "strength": "very_strong" if abs(val) > 0.9
                                    else "strong" if abs(val) > 0.7
                                    else "moderate",
                        "direction": "positive" if val > 0 else "negative"
                    })
        high_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return {"matrix": corr.to_dict(), "high_correlations": high_corr[:10]}

    def detect_outliers(self, df: pd.DataFrame, columns: list = None) -> dict:
        """IQR-based outlier detection for numeric columns."""
        cols = columns or list(df.select_dtypes(include='number').columns)
        results = {}
        for col in cols:
            if col not in df.columns:
                continue
            clean = df[col].dropna()
            if len(clean) == 0:
                continue
            Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask = (df[col] < lower) | (df[col] > upper)
            cnt = int(mask.sum())
            results[col] = {
                "count": cnt,
                "percentage": round(float(cnt / max(len(df), 1) * 100), 2),
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
                "min_outlier": round(float(df.loc[mask, col].min()), 3) if cnt > 0 else None,
                "max_outlier": round(float(df.loc[mask, col].max()), 3) if cnt > 0 else None
            }
        return results

    def scan_pii(self, df: pd.DataFrame) -> list:
        """Regex-based PII scanner for email, phone, CNIC, credit card, IP."""
        import re
        patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone_pk": r'(\+92|0)?[0-9]{10,11}',
            "cnic": r'[0-9]{5}-[0-9]{7}-[0-9]',
            "credit_card": r'\b(?:\d[ -]?){13,16}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        found = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().astype(str)
            for pii_type, pattern in patterns.items():
                try:
                    matches = int(sample.str.contains(pattern, regex=True, na=False).sum())
                    if matches > 0:
                        found.append({
                            "column": col, "pii_type": pii_type,
                            "count": matches,
                            "risk": "high" if pii_type in ["credit_card", "cnic"] else "medium"
                        })
                except Exception:
                    pass
        return found

    # ── CLEANING TOOLS ──────────────────────────────────────────────────────

    def remove_duplicates(self, df: pd.DataFrame):
        before = len(df)
        df_new = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df_new)
        code = "df = df.drop_duplicates().reset_index(drop=True)"
        return df_new, f"Removed {removed} duplicate rows", removed, code

    def fill_missing(self, df: pd.DataFrame, column: str = None, method: str = "auto"):
        df_new = df.copy()
        filled = 0
        code_lines = []
        cols = [column] if (column and column in df.columns) else list(df.columns)

        for col in cols:
            missing = int(df_new[col].isnull().sum())
            if missing == 0:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df_new[col])
            m = method
            if m == "auto":
                m = "mean" if is_numeric else "mode"

            try:
                if m == "mean" and is_numeric:
                    val = df_new[col].mean()
                    df_new[col] = df_new[col].fillna(val)
                    code_lines.append(f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)")
                elif m == "median" and is_numeric:
                    val = df_new[col].median()
                    df_new[col] = df_new[col].fillna(val)
                    code_lines.append(f"df['{col}'].fillna(df['{col}'].median(), inplace=True)")
                elif m == "mode":
                    mode_val = df_new[col].mode()
                    if len(mode_val) > 0:
                        df_new[col] = df_new[col].fillna(mode_val.iloc[0])
                        code_lines.append(f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)")
                elif m == "ffill":
                    df_new[col] = df_new[col].ffill().bfill()
                    code_lines.append(f"df['{col}'] = df['{col}'].ffill().bfill()")
                elif m == "knn":
                    try:
                        from sklearn.impute import KNNImputer
                        num_cols = list(df_new.select_dtypes(include='number').columns)
                        if num_cols:
                            imputer = KNNImputer(n_neighbors=min(5, len(df_new) - 1))
                            df_new[num_cols] = imputer.fit_transform(df_new[num_cols])
                            code_lines.append(
                                "from sklearn.impute import KNNImputer\n"
                                "imputer = KNNImputer(n_neighbors=5)\n"
                                "df[numeric_cols] = imputer.fit_transform(df[numeric_cols])"
                            )
                            filled += missing
                            break
                    except Exception:
                        df_new[col] = df_new[col].fillna(df_new[col].mean() if is_numeric else df_new[col].mode().iloc[0])
                else:
                    # constant / fallback
                    if is_numeric:
                        df_new[col] = df_new[col].fillna(df_new[col].median())
                    else:
                        mode_v = df_new[col].mode()
                        if len(mode_v):
                            df_new[col] = df_new[col].fillna(mode_v.iloc[0])
                filled += missing
            except Exception as e:
                # Skip column on error
                pass

        code = "\n".join(code_lines)
        return df_new, f"Filled {filled} missing values", filled, code

    def fix_outliers(self, df: pd.DataFrame, column: str = None, method: str = "cap"):
        df_new = df.copy()
        affected = 0
        code_lines = []
        cols = [column] if (column and column in df.columns) else list(df.select_dtypes(include='number').columns)

        for col in cols:
            clean = df_new[col].dropna()
            if len(clean) == 0:
                continue
            Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask = (df_new[col] < lower) | (df_new[col] > upper)
            col_affected = int(mask.sum())

            if method == "cap":
                df_new[col] = df_new[col].clip(lower, upper)
                code_lines.append(f"df['{col}'] = df['{col}'].clip({round(lower, 3)}, {round(upper, 3)})")
            elif method == "remove":
                df_new = df_new[~mask]
            elif method == "winsorize":
                try:
                    from scipy.stats.mstats import winsorize as sp_winsorize
                    df_new[col] = list(sp_winsorize(df_new[col].dropna(), limits=[0.05, 0.05]))
                except Exception:
                    df_new[col] = df_new[col].clip(lower, upper)
                code_lines.append(f"# winsorize '{col}' at 5% limits")

            affected += col_affected

        code = "\n".join(code_lines)
        return df_new, f"Fixed {affected} outliers using {method}", affected, code

    def standardize_text(self, df: pd.DataFrame, column: str = None):
        df_new = df.copy()
        fixed = 0
        code_lines = []
        cols = [column] if (column and column in df.columns) else list(df.select_dtypes(include='object').columns)

        for col in cols:
            if col not in df_new.columns:
                continue
            before = df_new[col].copy()
            try:
                df_new[col] = df_new[col].astype(str).str.strip()
                df_new[col] = df_new[col].str.replace(r'\s+', ' ', regex=True)
                df_new[col] = df_new[col].replace('nan', np.nan)
                changed = int((df_new[col] != before).sum())
                fixed += changed
                code_lines.append(f"df['{col}'] = df['{col}'].str.strip().str.replace(r'\\s+', ' ', regex=True)")
            except Exception:
                pass

        code = "\n".join(code_lines)
        return df_new, f"Standardized text in {len(cols)} columns ({fixed} cells changed)", fixed, code

    def convert_types(self, df: pd.DataFrame):
        df_new = df.copy()
        converted = []
        code_lines = []

        for col in df_new.select_dtypes(include='object').columns:
            # Try numeric
            num = pd.to_numeric(df_new[col], errors='coerce')
            if num.notna().sum() / max(len(df_new), 1) > 0.9:
                df_new[col] = num
                converted.append(col)
                code_lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
                continue
            # Try datetime
            try:
                dt = pd.to_datetime(df_new[col], errors='coerce', infer_datetime_format=True)
                if dt.notna().sum() / max(len(df_new), 1) > 0.8:
                    df_new[col] = dt
                    converted.append(col)
                    code_lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
            except Exception:
                pass

        code = "\n".join(code_lines)
        msg = f"Converted {len(converted)} columns: {', '.join(converted)}" if converted else "No type conversions needed"
        return df_new, msg, len(converted), code

    def remove_empty_columns(self, df: pd.DataFrame):
        threshold = 0.8
        empty_cols = [col for col in df.columns if df[col].isnull().sum() / max(len(df), 1) > threshold]
        df_new = df.drop(columns=empty_cols)
        code = f"df = df.drop(columns={empty_cols})"
        return df_new, f"Removed {len(empty_cols)} nearly-empty columns: {empty_cols}", len(empty_cols), code

    def remove_constant_columns(self, df: pd.DataFrame):
        const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        df_new = df.drop(columns=const_cols)
        code = f"df = df.drop(columns={const_cols})"
        return df_new, f"Removed {len(const_cols)} constant columns: {const_cols}", len(const_cols), code

    # ── FEATURE ENGINEERING ─────────────────────────────────────────────────

    def extract_date_features(self, df: pd.DataFrame, column: str):
        df_new = df.copy()
        code_lines = [f"df['{column}'] = pd.to_datetime(df['{column}'])"]
        try:
            df_new[column] = pd.to_datetime(df_new[column], errors='coerce')
            feats = {
                f'{column}_year': df_new[column].dt.year,
                f'{column}_month': df_new[column].dt.month,
                f'{column}_day': df_new[column].dt.day,
                f'{column}_dayofweek': df_new[column].dt.dayofweek,
                f'{column}_quarter': df_new[column].dt.quarter,
                f'{column}_is_weekend': df_new[column].dt.dayofweek.isin([5, 6]).astype(int)
            }
            for name, series in feats.items():
                df_new[name] = series
                code_lines.append(f"df['{name}'] = df['{column}'].dt.{name.replace(column + '_', '')}")
            added = list(feats.keys())
        except Exception as e:
            return df, f"Could not extract date features: {e}", 0, ""
        code = "\n".join(code_lines)
        return df_new, f"Extracted {len(added)} date features from '{column}'", len(added), code

    def encode_categorical(self, df: pd.DataFrame, column: str = None, method: str = "auto"):
        df_new = df.copy()
        encoded = []
        code_lines = []
        cols = [column] if (column and column in df.columns) else list(df.select_dtypes(include='object').columns)

        for col in cols:
            if col not in df_new.columns:
                continue
            unique_count = df_new[col].nunique()
            m = method
            if m == "auto":
                m = "onehot" if unique_count <= 10 else "frequency"

            try:
                if m == "onehot" and unique_count <= 15:
                    dummies = pd.get_dummies(df_new[col], prefix=col, drop_first=True, dtype=int)
                    df_new = pd.concat([df_new.drop(columns=[col]), dummies], axis=1)
                    code_lines.append(f"df = pd.concat([df.drop(columns=['{col}']), pd.get_dummies(df['{col}'], prefix='{col}', drop_first=True)], axis=1)")
                elif m == "label":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_new[col] = le.fit_transform(df_new[col].astype(str))
                    code_lines.append(f"from sklearn.preprocessing import LabelEncoder\ndf['{col}'] = LabelEncoder().fit_transform(df['{col}'])")
                elif m == "frequency":
                    freq = df_new[col].value_counts(normalize=True)
                    df_new[col] = df_new[col].map(freq)
                    code_lines.append(f"freq = df['{col}'].value_counts(normalize=True)\ndf['{col}'] = df['{col}'].map(freq)")
                else:
                    # fallback label
                    from sklearn.preprocessing import LabelEncoder
                    df_new[col] = LabelEncoder().fit_transform(df_new[col].astype(str))
                encoded.append(col)
            except Exception:
                pass

        code = "\n".join(code_lines)
        return df_new, f"Encoded {len(encoded)} categorical columns", len(encoded), code

    def scale_features(self, df: pd.DataFrame, method: str = "robust", columns: list = None):
        df_new = df.copy()
        cols = columns or list(df.select_dtypes(include='number').columns)
        if not cols:
            return df_new, "No numeric columns to scale", 0, ""

        try:
            if method == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                label = "StandardScaler"
            elif method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                label = "MinMaxScaler"
            else:
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                label = "RobustScaler"

            df_new[cols] = scaler.fit_transform(df_new[cols])
            code = (f"from sklearn.preprocessing import {label}\n"
                    f"scaler = {label}()\n"
                    f"df[numeric_cols] = scaler.fit_transform(df[numeric_cols])")
            return df_new, f"Scaled {len(cols)} features with {label}", len(cols), code
        except Exception as e:
            return df, f"Scaling failed: {e}", 0, ""

    # ── ML TOOLS ────────────────────────────────────────────────────────────

    def auto_train(self, df: pd.DataFrame, target_col: str, task_type: str = None) -> dict:
        """Train multiple ML models and return ranked leaderboard."""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder

        df_ml = df.dropna(subset=[target_col]).copy() if target_col in df.columns else df.dropna().copy()

        if target_col not in df_ml.columns:
            return {"error": f"Target column '{target_col}' not found"}
        if len(df_ml) < 20:
            return {"error": "Not enough rows to train (need at least 20 non-null rows)"}

        y = df_ml[target_col].copy()
        X = df_ml.drop(columns=[target_col]).copy()

        # Encode remaining object columns in X
        for col in X.select_dtypes(include='object').columns:
            try:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            except Exception:
                X = X.drop(columns=[col])

        X = X.select_dtypes(include='number').fillna(X.select_dtypes(include='number').median())

        # Infer task type
        if task_type is None:
            task_type = "classification" if (y.dtype == 'object' or y.nunique() < 15) else "regression"

        if task_type == "classification":
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
        else:
            y = pd.to_numeric(y, errors='coerce').fillna(y.median() if pd.api.types.is_numeric_dtype(y) else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models
        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50),
            }
            try:
                from xgboost import XGBClassifier
                models["XGBoost"] = XGBClassifier(random_state=42, verbosity=0, n_estimators=50)
            except ImportError:
                pass
            try:
                from lightgbm import LGBMClassifier
                models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1, n_estimators=50)
            except ImportError:
                pass
            scoring = "f1_weighted"
        else:
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=50),
            }
            try:
                from xgboost import XGBRegressor
                models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0, n_estimators=50)
            except ImportError:
                pass
            try:
                from lightgbm import LGBMRegressor
                models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1, n_estimators=50)
            except ImportError:
                pass
            scoring = "r2"

        leaderboard = []
        for name, model in models.items():
            try:
                t0 = time.time()
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring=scoring, n_jobs=-1)
                model.fit(X_train, y_train)
                elapsed = round((time.time() - t0) * 1000)
                y_pred = model.predict(X_test)

                if task_type == "classification":
                    from sklearn.metrics import f1_score, accuracy_score
                    metrics = {
                        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                        "f1_weighted": round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                        "cv_mean": round(float(cv_scores.mean()), 4),
                        "cv_std": round(float(cv_scores.std()), 4)
                    }
                else:
                    from sklearn.metrics import r2_score, mean_absolute_error
                    metrics = {
                        "r2": round(float(r2_score(y_test, y_pred)), 4),
                        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                        "cv_mean": round(float(cv_scores.mean()), 4),
                        "cv_std": round(float(cv_scores.std()), 4)
                    }

                feat_imp = None
                if hasattr(model, 'feature_importances_'):
                    pairs = sorted(
                        zip(X.columns.tolist(), model.feature_importances_.tolist()),
                        key=lambda x: x[1], reverse=True
                    )[:10]
                    feat_imp = [{"feature": f, "importance": round(i, 4)} for f, i in pairs]

                leaderboard.append({
                    "model": name, "metrics": metrics,
                    "training_time_ms": elapsed,
                    "feature_importance": feat_imp
                })
            except Exception as e:
                leaderboard.append({"model": name, "metrics": {}, "error": str(e)})

        primary = "f1_weighted" if task_type == "classification" else "r2"
        leaderboard.sort(key=lambda x: x["metrics"].get(primary, -999), reverse=True)

        code = (
            f"from sklearn.ensemble import RandomForest{'Classifier' if task_type == 'classification' else 'Regressor'}\n"
            f"from sklearn.model_selection import train_test_split\n"
            f"X = df.drop(columns=['{target_col}'])\n"
            f"y = df['{target_col}']\n"
            f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            f"model = RandomForest{'Classifier' if task_type == 'classification' else 'Regressor'}(n_estimators=100, random_state=42)\n"
            f"model.fit(X_train, y_train)\n"
            f"print('Score:', model.score(X_test, y_test))"
        )
        return {
            "leaderboard": leaderboard,
            "best_model": leaderboard[0]["model"] if leaderboard else None,
            "task_type": task_type,
            "target_col": target_col,
            "features_used": X.columns.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "code": code
        }

    def generate_insights(self, df: pd.DataFrame, analysis_results: dict, dataset_type: str) -> list:
        """Generate business insights using Groq AI."""
        from services.agent_brain import agent_brain
        df_stats = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().round(2).to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return agent_brain.generate_business_insights(df_stats, analysis_results, dataset_type)

    def export_data(self, df: pd.DataFrame, fmt: str = "csv"):
        """Export cleaned DataFrame in CSV, Excel, or JSON format."""
        if fmt == "csv":
            output = df.to_csv(index=False)
            code = "df.to_csv('cleaned_data.csv', index=False)"
        elif fmt == "excel":
            from io import BytesIO
            buf = BytesIO()
            df.to_excel(buf, index=False, engine='openpyxl')
            output = buf.getvalue().hex()  # serialise for JSON
            code = "df.to_excel('cleaned_data.xlsx', index=False)"
        else:
            output = df.to_json(orient='records', indent=2)
            code = "df.to_json('cleaned_data.json', orient='records', indent=2)"
        return None, f"Data ready for export as {fmt.upper()}", 0, code


tools = DataScienceTools()
