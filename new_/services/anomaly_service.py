"""Anomaly Detection Engine Service."""

import re
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def detect_all_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect statistical, pattern, text, date, and business rule anomalies across dataframe."""
    anomalies: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return anomalies

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # ─────────────────────────────────────────────────────────────
    # 1. STATISTICAL ANOMALIES
    # ─────────────────────────────────────────────────────────────
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue

        std_val = float(s.std())
        mean_val = float(s.mean())

        # Z-score > 3
        if std_val > 1e-9:
            z_scores = (s - mean_val).abs() / std_val
            z_outliers = z_scores[z_scores > 3.0]
            if len(z_outliers) > 0:
                out_idx = z_outliers.index.tolist()
                anomalies.append({
                    "type": "statistical_zscore",
                    "severity": "critical" if len(z_outliers) / len(s) > 0.05 else "warning",
                    "column": str(col),
                    "row_indices": out_idx[:100],
                    "count": int(len(z_outliers)),
                    "description": f"Found {len(z_outliers)} extreme outliers with |Z-score| > 3",
                    "example_value": float(s.loc[out_idx[0]]),
                    "recommended_action": "Winsorize or cap at 3 standard deviations"
                })

        # IQR method (mild: 1.5-3x IQR, extreme: >3x IQR)
        q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q75 - q25
        if iqr > 0:
            ext_lower = q25 - 3.0 * iqr
            ext_upper = q75 + 3.0 * iqr
            ext_mask = (s < ext_lower) | (s > ext_upper)
            if ext_mask.sum() > 0:
                ext_idx = s[ext_mask].index.tolist()
                anomalies.append({
                    "type": "statistical_iqr_extreme",
                    "severity": "critical",
                    "column": str(col),
                    "row_indices": ext_idx[:100],
                    "count": int(len(ext_idx)),
                    "description": f"Found {len(ext_idx)} values beyond 3x IQR boundaries",
                    "example_value": float(s.loc[ext_idx[0]]),
                    "recommended_action": "Cap at IQR bounds or investigate sensor/entry error"
                })

    # Mahalanobis multivariate outliers
    if len(numeric_cols) >= 2 and n_rows > len(numeric_cols) * 2:
        try:
            clean_num = df[numeric_cols].dropna()
            if len(clean_num) > len(numeric_cols) + 1:
                cov = np.cov(clean_num.values, rowvar=False)
                inv_cov = np.linalg.pinv(cov)
                diff = clean_num.values - clean_num.mean().values
                left = np.dot(diff, inv_cov)
                mahal = np.sqrt(np.sum(left * diff, axis=1))
                cutoff = float(np.sqrt(stats.chi2.ppf(0.999, df=len(numeric_cols))))
                multi_outliers = np.where(mahal > cutoff)[0]
                if len(multi_outliers) > 0:
                    real_idx = clean_num.index[multi_outliers].tolist()
                    anomalies.append({
                        "type": "multivariate_mahalanobis",
                        "severity": "warning",
                        "column": ", ".join(numeric_cols[:3]),
                        "row_indices": real_idx[:100],
                        "count": int(len(real_idx)),
                        "description": f"Detected {len(real_idx)} multivariate outliers using Mahalanobis distance (p<0.001)",
                        "example_value": f"Row index {real_idx[0]}",
                        "recommended_action": "Review multi-column combination for data entry corruptions"
                    })
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # 2. PATTERN ANOMALIES
    # ─────────────────────────────────────────────────────────────
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        std_val = float(s.std())
        if std_val > 1e-9:
            diffs = s.diff().abs()
            sudden_jumps = diffs[diffs > 3.0 * std_val]
            if len(sudden_jumps) > 0:
                anomalies.append({
                    "type": "pattern_sudden_jump",
                    "severity": "warning",
                    "column": str(col),
                    "row_indices": sudden_jumps.index.tolist()[:100],
                    "count": int(len(sudden_jumps)),
                    "description": f"Detected {len(sudden_jumps)} abrupt value jumps (> 3*std difference)",
                    "example_value": float(s.loc[sudden_jumps.index[0]]),
                    "recommended_action": "Smooth sequential jumps or check sensor resets"
                })

        # Repeated sequences: same value repeated 10+ times in a row
        consec_counts = (s != s.shift()).cumsum()
        streak_lens = s.groupby(consec_counts).transform("count")
        long_streaks = s[streak_lens >= 10]
        if len(long_streaks) > 0:
            streak_idx = long_streaks.index.tolist()
            anomalies.append({
                "type": "pattern_repeated_sequence",
                "severity": "warning",
                "column": str(col),
                "row_indices": streak_idx[:100],
                "count": int(len(streak_idx)),
                "description": f"Detected repeating sequence of identical value ({s.loc[streak_idx[0]]}) over 10+ consecutive rows",
                "example_value": float(s.loc[streak_idx[0]]),
                "recommended_action": "Check for frozen data ingestion or placeholder fills"
            })

        # Round number bias: 80%+ values are exact multiples of 10 or 100
        if len(s) >= 20:
            int_candidates = s[(s == s.round()) & (s > 0)]
            if len(int_candidates) > 0.7 * len(s):
                round_count = (int_candidates % 10 == 0).sum()
                if round_count / len(int_candidates) >= 0.8:
                    anomalies.append({
                        "type": "pattern_round_number_bias",
                        "severity": "info",
                        "column": str(col),
                        "row_indices": int_candidates[int_candidates % 10 == 0].index.tolist()[:50],
                        "count": int(round_count),
                        "description": f"Over 80% of values are rounded numbers (multiples of 10)",
                        "example_value": float(int_candidates.iloc[0]),
                        "recommended_action": "Note potential manual estimations or survey discretization"
                    })

        # Suspicious zeros: 0 might mean missing if all non-zero values are large
        zeros = s[s == 0]
        non_zeros = s[s > 0]
        if len(zeros) > 0 and len(non_zeros) > 0:
            if non_zeros.min() > 50 and (len(zeros) / len(s)) > 0.03:
                anomalies.append({
                    "type": "pattern_suspicious_zeros",
                    "severity": "warning",
                    "column": str(col),
                    "row_indices": zeros.index.tolist()[:100],
                    "count": int(len(zeros)),
                    "description": f"{len(zeros)} zero values found while minimum positive value is {non_zeros.min()} (0 likely represents missing)",
                    "example_value": 0,
                    "recommended_action": "Convert zeros to NaN and impute"
                })

    # ─────────────────────────────────────────────────────────────
    # 3. TEXT ANOMALIES
    # ─────────────────────────────────────────────────────────────
    for col in object_cols:
        s = df[col].dropna().astype(str)
        if len(s) < 5:
            continue

        lengths = s.str.len()
        mean_len = float(lengths.mean())
        std_len = float(lengths.std())

        # Extremely long values
        if std_len > 2:
            cutoff_len = mean_len + 3.0 * std_len
            extreme_lens = s[lengths > cutoff_len]
            if len(extreme_lens) > 0:
                anomalies.append({
                    "type": "text_extreme_length",
                    "severity": "warning",
                    "column": str(col),
                    "row_indices": extreme_lens.index.tolist()[:100],
                    "count": int(len(extreme_lens)),
                    "description": f"{len(extreme_lens)} text entries exceed mean length by >3 standard deviations",
                    "example_value": str(extreme_lens.iloc[0])[:60] + "...",
                    "recommended_action": "Truncate text or extract key structured attributes"
                })

        # Copy-pasted long values repeated multiple times
        long_strings = s[lengths > 40]
        if len(long_strings) > 0:
            repeats = long_strings.value_counts()
            top_repeats = repeats[repeats >= 5]
            if len(top_repeats) > 0:
                val_str = top_repeats.index[0]
                matching_idx = s[s == val_str].index.tolist()
                anomalies.append({
                    "type": "text_copy_paste_anomaly",
                    "severity": "info",
                    "column": str(col),
                    "row_indices": matching_idx[:100],
                    "count": int(len(matching_idx)),
                    "description": f"Identical long string repeated {len(matching_idx)} times across records",
                    "example_value": str(val_str)[:60] + "...",
                    "recommended_action": "Verify if template text or default fallback was auto-injected"
                })

        # Mixed scripts / languages
        non_ascii_mask = s.apply(lambda x: any(ord(c) > 127 for c in x))
        ascii_mask = ~non_ascii_mask
        if non_ascii_mask.sum() > 0 and ascii_mask.sum() > 0:
            if non_ascii_mask.sum() / len(s) > 0.05 and ascii_mask.sum() / len(s) > 0.05:
                non_asc_idx = s[non_ascii_mask].index.tolist()
                anomalies.append({
                    "type": "text_mixed_scripts",
                    "severity": "info",
                    "column": str(col),
                    "row_indices": non_asc_idx[:50],
                    "count": int(len(non_asc_idx)),
                    "description": f"Mixed character sets (ASCII and non-ASCII characters) detected in column",
                    "example_value": str(s.loc[non_asc_idx[0]]),
                    "recommended_action": "Normalize unicode or transcode to consistent encoding"
                })

    # ─────────────────────────────────────────────────────────────
    # 4. DATE ANOMALIES
    # ─────────────────────────────────────────────────────────────
    now = datetime.now()
    for col in df.columns:
        # Check if date
        is_date = pd.api.types.is_datetime64_any_dtype(df[col])
        parsed = None
        if not is_date and any(d in str(col).lower() for d in ["date", "time", "dob", "created", "timestamp"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0.5 * len(df):
                    is_date = True
            except Exception:
                pass
        elif is_date:
            parsed = df[col]

        if is_date and parsed is not None:
            valid_dates = parsed.dropna()
            if len(valid_dates) > 0:
                # Future dates
                future_mask = valid_dates > now
                if future_mask.sum() > 0 and not any(f in str(col).lower() for f in ["due", "expiry", "target", "future"]):
                    f_idx = valid_dates[future_mask].index.tolist()
                    anomalies.append({
                        "type": "date_future",
                        "severity": "critical",
                        "column": str(col),
                        "row_indices": f_idx[:100],
                        "count": int(len(f_idx)),
                        "description": f"Detected {len(f_idx)} future dates past current time",
                        "example_value": str(valid_dates.loc[f_idx[0]].date()),
                        "recommended_action": "Cap date at current timestamp or verify year component"
                    })

                # Very old dates (< 1900)
                try:
                    old_mask = valid_dates < pd.Timestamp("1900-01-01")
                    if old_mask.sum() > 0:
                        o_idx = valid_dates[old_mask].index.tolist()
                        anomalies.append({
                            "type": "date_historic_old",
                            "severity": "warning",
                            "column": str(col),
                            "row_indices": o_idx[:100],
                            "count": int(len(o_idx)),
                            "description": f"Found {len(o_idx)} dates prior to year 1900",
                            "example_value": str(valid_dates.loc[o_idx[0]].date()),
                            "recommended_action": "Correct typo or replace placeholder default dates (e.g. 1970/1899)"
                        })
                except Exception:
                    pass

                # Date jumps (> 1 year gap)
                sorted_dates = valid_dates.sort_values()
                date_diffs = sorted_dates.diff().dt.days
                huge_jumps = date_diffs[date_diffs > 365]
                if len(huge_jumps) > 0:
                    j_idx = sorted_dates.loc[huge_jumps.index].index.tolist()
                    anomalies.append({
                        "type": "date_gap_jump",
                        "severity": "warning",
                        "column": str(col),
                        "row_indices": j_idx[:50],
                        "count": int(len(huge_jumps)),
                        "description": f"Found {len(huge_jumps)} chronological gaps exceeding 365 days",
                        "example_value": f"{huge_jumps.iloc[0]} days gap",
                        "recommended_action": "Inspect for missing periods in dataset collection"
                    })

                # Weekend dates in business context
                if any(b in str(col).lower() for b in ["work", "trade", "invoice", "payroll", "business"]):
                    weekends = valid_dates[valid_dates.dt.dayofweek >= 5]
                    if len(weekends) > 0:
                        anomalies.append({
                            "type": "date_weekend_business",
                            "severity": "info",
                            "column": str(col),
                            "row_indices": weekends.index.tolist()[:50],
                            "count": int(len(weekends)),
                            "description": f"Found {len(weekends)} weekend dates in business column",
                            "example_value": str(weekends.iloc[0].date()),
                            "recommended_action": "Shift weekend records to nearest preceding business day"
                        })

    # ─────────────────────────────────────────────────────────────
    # 5. BUSINESS RULE ANOMALIES
    # ─────────────────────────────────────────────────────────────
    for col in numeric_cols:
        col_lower = str(col).lower()
        s = df[col].dropna()

        # Negative values in strictly positive fields
        if any(w in col_lower for w in ["age", "price", "salary", "cost", "revenue", "qty", "quantity", "count", "height", "weight"]):
            negs = s[s < 0]
            if len(negs) > 0:
                anomalies.append({
                    "type": "business_negative_value",
                    "severity": "critical",
                    "column": str(col),
                    "row_indices": negs.index.tolist()[:100],
                    "count": int(len(negs)),
                    "description": f"Found {len(negs)} illegal negative values in positive domain column '{col}'",
                    "example_value": float(negs.iloc[0]),
                    "recommended_action": "Take absolute value or replace with column median"
                })

        # Percentage values > 100
        if any(w in col_lower for w in ["percent", "pct", "rate", "probability", "share"]):
            over100 = s[s > 100]
            if len(over100) > 0:
                anomalies.append({
                    "type": "business_percentage_over_100",
                    "severity": "critical",
                    "column": str(col),
                    "row_indices": over100.index.tolist()[:100],
                    "count": int(len(over100)),
                    "description": f"Found {len(over100)} percentage values exceeding 100%",
                    "example_value": float(over100.iloc[0]),
                    "recommended_action": "Cap values at 100 or check if scaled by factor of 100"
                })

    # ID columns with duplicates
    for col in df.columns:
        col_lower = str(col).lower()
        if any(w in col_lower for w in ["id", "uuid", "key", "identifier", "code", "account_no", "user_id"]) and not col_lower.endswith("_idx"):
            dups = df[col][df[col].duplicated(keep=False)].dropna()
            if len(dups) > 0:
                anomalies.append({
                    "type": "business_duplicate_identifier",
                    "severity": "critical",
                    "column": str(col),
                    "row_indices": dups.index.tolist()[:100],
                    "count": int(len(dups)),
                    "description": f"Unique ID column '{col}' contains {len(dups)} duplicate entries",
                    "example_value": str(dups.iloc[0]),
                    "recommended_action": "Deduplicate rows or regenerate unique primary identifiers"
                })

    # Invalid Emails
    email_pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    for col in object_cols:
        col_lower = str(col).lower()
        s = df[col].dropna().astype(str)
        if "email" in col_lower or s.str.contains("@").sum() > 0.4 * len(s):
            invalid = s[~s.str.match(email_pattern)]
            if len(invalid) > 0:
                anomalies.append({
                    "type": "business_invalid_email",
                    "severity": "warning",
                    "column": str(col),
                    "row_indices": invalid.index.tolist()[:100],
                    "count": int(len(invalid)),
                    "description": f"Detected {len(invalid)} malformed email addresses",
                    "example_value": str(invalid.iloc[0]),
                    "recommended_action": "Clean whitespace or set invalid email entries to NULL"
                })

    return anomalies


def fix_all_anomalies(df: pd.DataFrame) -> (pd.DataFrame, int, List[str]):
    """Automatically fix all detected anomalies safely and return updated df, affected rows count, and fix log."""
    new_df = df.copy()
    anomalies = detect_all_anomalies(new_df)
    fixes_applied: List[str] = []
    affected_indices = set()

    for anom in anomalies:
        col = anom.get("column")
        a_type = anom.get("type", "")

        if a_type == "business_negative_value" and col in new_df.columns:
            mask = new_df[col] < 0
            count = int(mask.sum())
            if count > 0:
                affected_indices.update(new_df[mask].index.tolist())
                # Replace with abs value
                new_df[col] = new_df[col].abs()
                fixes_applied.append(f"Converted {count} negative values to positive in '{col}'")

        elif a_type == "business_percentage_over_100" and col in new_df.columns:
            mask = new_df[col] > 100
            count = int(mask.sum())
            if count > 0:
                affected_indices.update(new_df[mask].index.tolist())
                new_df.loc[mask, col] = 100.0
                fixes_applied.append(f"Capped {count} percentage values at 100% in '{col}'")

        elif a_type in ["statistical_zscore", "statistical_iqr_extreme"] and col in new_df.columns:
            s = new_df[col].dropna()
            if len(s) > 5:
                q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q75 - q25
                lower, upper = q25 - 2.5 * iqr, q75 + 2.5 * iqr
                mask = (new_df[col] < lower) | (new_df[col] > upper)
                count = int(mask.sum())
                if count > 0:
                    affected_indices.update(new_df[mask].index.tolist())
                    new_df[col] = new_df[col].clip(lower=lower, upper=upper)
                    fixes_applied.append(f"Winsorized/capped {count} extreme outliers in '{col}'")

        elif a_type == "pattern_suspicious_zeros" and col in new_df.columns:
            mask = new_df[col] == 0
            count = int(mask.sum())
            if count > 0:
                affected_indices.update(new_df[mask].index.tolist())
                med = float(new_df.loc[new_df[col] > 0, col].median())
                new_df.loc[mask, col] = med
                fixes_applied.append(f"Replaced {count} placeholder zeros with median ({med:.2f}) in '{col}'")

        elif a_type == "business_invalid_email" and col in new_df.columns:
            email_pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
            s = new_df[col].astype(str)
            mask = ~s.str.match(email_pattern) & new_df[col].notna()
            count = int(mask.sum())
            if count > 0:
                affected_indices.update(new_df[mask].index.tolist())
                new_df.loc[mask, col] = np.nan
                fixes_applied.append(f"Cleared {count} invalid email entries in '{col}' to NaN")

        elif a_type == "text_extreme_length" and col in new_df.columns:
            s = new_df[col].astype(str)
            lens = s.str.len()
            cutoff = int(lens.mean() + 3 * lens.std())
            mask = lens > cutoff
            count = int(mask.sum())
            if count > 0:
                affected_indices.update(new_df[mask].index.tolist())
                new_df.loc[mask, col] = s[mask].str.slice(0, cutoff)
                fixes_applied.append(f"Truncated {count} excessively long text records in '{col}' to {cutoff} chars")

        elif a_type == "business_duplicate_identifier" and col in new_df.columns:
            dups = new_df[col].duplicated()
            count = int(dups.sum())
            if count > 0:
                affected_indices.update(new_df[dups].index.tolist())
                new_df.loc[dups, col] = new_df.loc[dups, col].astype(str) + "_dup_" + new_df.index[dups].astype(str)
                fixes_applied.append(f"Deduplicated {count} duplicate IDs in '{col}'")

        elif a_type == "pattern_sudden_jump" and col in new_df.columns:
            s = new_df[col].dropna()
            std_val = float(s.std())
            if std_val > 1e-9:
                diffs = s.diff().abs()
                jump_mask = diffs > 3.0 * std_val
                count = int(jump_mask.sum())
                if count > 0:
                    affected_indices.update(new_df.loc[jump_mask.index[jump_mask]].index.tolist())
                    new_df[col] = new_df[col].rolling(3, min_periods=1, center=True).median()
                    fixes_applied.append(f"Smoothed {count} sudden value jumps in '{col}'")

    return new_df, len(affected_indices), fixes_applied
