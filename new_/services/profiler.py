"""Smart Data Profiling Service & ReportLab PDF Generator."""

import io
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from services.monitor import calculate_quality_metrics
from services.recommender import generate_smart_recommendations

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


def _sanitize(val):
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    return str(val)


def get_column_profile(df: pd.DataFrame, col_name: str) -> Dict[str, Any]:
    """Generate detailed profile for a single column."""
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found")

    n_rows = len(df)
    s = df[col_name]
    missing_count = int(s.isna().sum())
    missing_pct = round((missing_count / max(n_rows, 1)) * 100, 2)
    unique_count = int(s.nunique(dropna=True))
    unique_pct = round((unique_count / max(n_rows, 1)) * 100, 2)

    col_prof: Dict[str, Any] = {
        "name": col_name,
        "dtype": str(s.dtype),
        "total_rows": n_rows,
        "count": int(s.count()),
        "missing": missing_count,
        "missing_pct": missing_pct,
        "unique": unique_count,
        "unique_pct": unique_pct,
    }

    # Numeric Column Profile
    if pd.api.types.is_numeric_dtype(s):
        col_prof["kind"] = "numeric"
        clean = s.dropna()
        if len(clean) > 0:
            mean_v = float(clean.mean())
            med_v = float(clean.median())
            mode_series = clean.mode()
            mode_v = float(mode_series.iloc[0]) if not mode_series.empty else mean_v
            std_v = float(clean.std()) if len(clean) > 1 else 0.0
            var_v = float(clean.var()) if len(clean) > 1 else 0.0
            min_v = float(clean.min())
            max_v = float(clean.max())
            range_v = max_v - min_v

            q01, q05, q25, q50, q75, q95, q99 = [
                float(clean.quantile(q)) for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            ]
            iqr_v = q75 - q25
            lower_fence = q25 - 1.5 * iqr_v
            upper_fence = q75 + 1.5 * iqr_v
            outliers = clean[(clean < lower_fence) | (clean > upper_fence)]
            outlier_count = int(len(outliers))

            skew_v = float(clean.skew()) if len(clean) > 2 else 0.0
            kurt_v = float(clean.kurt()) if len(clean) > 3 else 0.0

            # Distribution type
            if abs(skew_v) < 0.5 and abs(kurt_v) < 1.0:
                dist_type = "Normal / Bell-shaped"
            elif skew_v > 1.0:
                dist_type = "Right-Skewed (Positive)"
            elif skew_v < -1.0:
                dist_type = "Left-Skewed (Negative)"
            elif abs(max_v - min_v) > 0 and abs(std_v / (max_v - min_v + 1e-9) - 0.288) < 0.05:
                dist_type = "Uniform"
            else:
                dist_type = "Multi-modal / Complex"

            # 10 Bins Histogram
            counts, bin_edges = np.histogram(clean, bins=min(10, max(2, len(clean.unique()))))
            hist_data = [
                {"bin_start": round(float(bin_edges[i]), 2),
                 "bin_end": round(float(bin_edges[i+1]), 2),
                 "count": int(counts[i])}
                for i in range(len(counts))
            ]

            col_prof.update({
                "mean": round(mean_v, 4),
                "median": round(med_v, 4),
                "mode": round(mode_v, 4),
                "std": round(std_v, 4),
                "variance": round(var_v, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                "range": round(range_v, 4),
                "iqr": round(iqr_v, 4),
                "skewness": round(skew_v, 4),
                "kurtosis": round(kurt_v, 4),
                "percentiles": {
                    "p1": round(q01, 4), "p5": round(q05, 4), "p25": round(q25, 4),
                    "p50": round(q50, 4), "p75": round(q75, 4), "p95": round(q95, 4),
                    "p99": round(q99, 4)
                },
                "outlier_count": outlier_count,
                "outlier_percentage": round((outlier_count / len(clean)) * 100, 2),
                "distribution_type": dist_type,
                "histogram": hist_data,
                "box_plot": {
                    "min": round(min_v, 4),
                    "q1": round(q25, 4),
                    "median": round(q50, 4),
                    "q3": round(q75, 4),
                    "max": round(max_v, 4),
                    "lower_whisker": round(max(min_v, lower_fence), 4),
                    "upper_whisker": round(min(max_v, upper_fence), 4),
                    "outliers_sample": [round(float(x), 2) for x in outliers.head(10).tolist()]
                }
            })

    # Date Column Profile
    elif pd.api.types.is_datetime64_any_dtype(s) or (
        s.dropna().astype(str).str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").sum() > 0.6 * n_rows
    ):
        col_prof["kind"] = "date"
        dates = pd.to_datetime(s, errors="coerce").dropna().sort_values()
        if len(dates) > 0:
            min_d = dates.iloc[0]
            max_d = dates.iloc[-1]
            diff_days = dates.diff().dt.days
            col_prof.update({
                "min_date": str(min_d.date()),
                "max_date": str(max_d.date()),
                "date_range_days": int((max_d - min_d).days),
                "most_common_year": int(dates.dt.year.mode().iloc[0]) if not dates.dt.year.empty else None,
                "most_common_month": int(dates.dt.month.mode().iloc[0]) if not dates.dt.month.empty else None,
                "most_common_day": int(dates.dt.day.mode().iloc[0]) if not dates.dt.day.empty else None,
                "date_gaps_detected": bool((diff_days > 365).any())
            })

    # Text / Categorical Column Profile
    else:
        col_prof["kind"] = "categorical"
        str_s = s.dropna().astype(str)
        if len(str_s) > 0:
            val_counts = str_s.value_counts().head(10)
            top_10 = [{"value": str(idx), "count": int(cnt)} for idx, cnt in val_counts.items()]
            lengths = str_s.str.len()
            col_prof.update({
                "top_values": top_10,
                "avg_length": round(float(lengths.mean()), 1),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "contains_numbers": bool(str_s.str.contains(r"\d").any()),
                "contains_special_chars": bool(str_s.str.contains(r"[^\w\s]").any()),
                "sample_values": str_s.drop_duplicates().head(5).tolist()
            })

    return col_prof


def generate_full_profile(df: pd.DataFrame, session_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate comprehensive 6-section profiling report."""
    if session_info is None:
        session_info = {}

    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)

    # 1. DATASET OVERVIEW
    mem_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
    quality_metrics = calculate_quality_metrics(df)

    dtypes_breakdown = {
        "numeric": int(len(df.select_dtypes(include=[np.number]).columns)),
        "text": int(len(df.select_dtypes(include=["object", "string"]).columns)),
        "date": int(len(df.select_dtypes(include=["datetime", "datetimetz"]).columns)),
        "boolean": int(len(df.select_dtypes(include=["bool"]).columns))
    }

    overview = {
        "file_name": session_info.get("file_name", "dataset.csv"),
        "total_rows": n_rows,
        "total_columns": n_cols,
        "total_cells": total_cells,
        "memory_mb": mem_mb,
        "file_size": f"{mem_mb} MB",
        "data_types_breakdown": dtypes_breakdown,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round((int(df.duplicated().sum()) / max(n_rows, 1)) * 100, 2),
        "missing_cells": quality_metrics["completeness"]["missing_cells"],
        "missing_pct": quality_metrics["completeness"]["missing_percentage"],
        "quality_score": quality_metrics["overall_score"],
        "grade": quality_metrics["grade"]
    }

    # 2. COLUMN PROFILES
    column_profiles = {col: get_column_profile(df, col) for col in df.columns}

    # 3. CORRELATIONS & MULTICOLLINEARITY & VIF
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix: Dict[str, Dict[str, float]] = {}
    top_correlated_pairs: List[Dict[str, Any]] = []
    multicollinearity_warnings: List[Dict[str, Any]] = []
    vif_dict: Dict[str, float] = {}

    if len(numeric_cols) >= 2:
        clean_num = df[numeric_cols].dropna()
        if len(clean_num) > 2:
            corr_df = clean_num.corr()
            for col1 in numeric_cols:
                correlation_matrix[col1] = {}
                for col2 in numeric_cols:
                    val = corr_df.loc[col1, col2]
                    correlation_matrix[col1][col2] = None if (math.isnan(val) or math.isinf(val)) else round(float(val), 3)

            # High pairs
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    r = corr_df.loc[c1, c2]
                    if not (math.isnan(r) or math.isinf(r)):
                        pairs.append({"col1": c1, "col2": c2, "correlation": round(float(r), 3), "abs_corr": abs(float(r))})

            pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
            top_correlated_pairs = pairs[:10]

            # Multicollinearity (> 0.85)
            for p in pairs:
                if p["abs_corr"] > 0.85:
                    multicollinearity_warnings.append({
                        "col1": p["col1"],
                        "col2": p["col2"],
                        "correlation": p["correlation"],
                        "warning": f"High collinearity ({p['correlation']:.2f}) between '{p['col1']}' and '{p['col2']}'"
                    })

            # Calculate VIF via auxiliary regressions
            if len(numeric_cols) > 2 and len(clean_num) > len(numeric_cols):
                for idx, target_c in enumerate(numeric_cols):
                    feature_cols = [c for c in numeric_cols if c != target_c]
                    try:
                        X = clean_num[feature_cols].values
                        y = clean_num[target_c].values
                        lr = LinearRegression().fit(X, y)
                        r2 = max(0.0, float(lr.score(X, y)))
                        vif = 999.9 if r2 >= 0.999 else round(1.0 / (1.0 - r2), 2)
                        vif_dict[target_c] = vif
                    except Exception:
                        vif_dict[target_c] = 1.0

    # 4. MISSING VALUES ANALYSIS
    missing_by_column = []
    missing_matrix = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        pct = round((cnt / max(n_rows, 1)) * 100, 2)
        missing_by_column.append({
            "column": col,
            "missing_count": cnt,
            "missing_pct": pct,
            "recommended_imputation": "KNN / Iterative" if pd.api.types.is_numeric_dtype(df[col]) else "Mode / Constant"
        })

    # MCAR/MAR/MNAR classification attempt
    mcar_assessment = "MCAR (Missing Completely at Random)"
    if quality_metrics["completeness"]["missing_cells"] > 0:
        # Check correlation of missingness indicators
        missing_indicators = df.isna().astype(int)
        corr_miss = missing_indicators.corr().abs()
        corr_vals = corr_miss.to_numpy(copy=True)
        if corr_vals.ndim == 2 and corr_vals.shape[0] == corr_vals.shape[1]:
            np.fill_diagonal(corr_vals, 0)
        max_miss_corr = float(np.nanmax(corr_vals)) if corr_vals.size > 0 else 0.0
        if max_miss_corr > 0.5:
            mcar_assessment = "MAR (Missing at Random - correlated patterns detected between missing columns)"
        elif max_miss_corr > 0.2:
            mcar_assessment = "Possible MNAR (Missing Not at Random - conditional dependencies)"

    # 5. OUTLIER ANALYSIS
    outlier_analysis = {
        "per_column": [],
        "overall_outlier_rows": 0
    }
    outlier_rows_set = set()
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 4:
            q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q75 - q25
            if iqr > 0:
                mask = (df[col] < q25 - 1.5 * iqr) | (df[col] > q75 + 1.5 * iqr)
                cnt = int(mask.sum())
                if cnt > 0:
                    outlier_rows_set.update(df[mask].index.tolist())
                    outliers_val = df.loc[mask, col]
                    outlier_analysis["per_column"].append({
                        "column": col,
                        "outlier_count": cnt,
                        "outlier_pct": round((cnt / max(n_rows, 1)) * 100, 2),
                        "min_outlier": round(float(outliers_val.min()), 2),
                        "max_outlier": round(float(outliers_val.max()), 2)
                    })
    outlier_analysis["overall_outlier_rows"] = len(outlier_rows_set)

    # 6. DATA QUALITY ISSUES
    issues = []
    if overview["duplicate_rows"] > 0:
        issues.append({
            "issue": f"{overview['duplicate_rows']} Duplicate Rows Detected",
            "severity": "critical" if overview["duplicate_pct"] > 5 else "warning",
            "recommended_fix": "Drop duplicate records from dataset"
        })
    if overview["missing_cells"] > 0:
        issues.append({
            "issue": f"{overview['missing_cells']} Missing Data Cells ({overview['missing_pct']}%)",
            "severity": "critical" if overview["missing_pct"] > 10 else "warning",
            "recommended_fix": "Apply KNN imputation for continuous features and mode for categoricals"
        })
    if outlier_analysis["overall_outlier_rows"] > 0:
        issues.append({
            "issue": f"{outlier_analysis['overall_outlier_rows']} Outlier Rows Identified via IQR",
            "severity": "warning",
            "recommended_fix": "Cap with Winsorization or truncate extreme anomalies"
        })
    if multicollinearity_warnings:
        issues.append({
            "issue": f"{len(multicollinearity_warnings)} High Multicollinearity Pair(s) (r > 0.85)",
            "severity": "warning",
            "recommended_fix": "Drop redundant redundant features to stabilize regression coefficients"
        })

    return {
        "overview": overview,
        "column_profiles": column_profiles,
        "correlations": {
            "matrix": correlation_matrix,
            "top_correlated_pairs": top_correlated_pairs,
            "multicollinearity_warnings": multicollinearity_warnings,
            "vif": vif_dict
        },
        "missing_analysis": {
            "by_column": missing_by_column,
            "pattern_classification": mcar_assessment,
            "total_missing_cells": overview["missing_cells"]
        },
        "outlier_analysis": outlier_analysis,
        "quality_issues": issues,
        "recommendations": generate_smart_recommendations(df)
    }


class NumberedCanvas(canvas.Canvas):
    """Canvas that tracks total page count for branding footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header (pages after cover)
        if self._pageNumber > 1:
            self.drawString(54, 750, "DataStudio AI — Intelligent Data Profiling Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY • GENERATED BY DATASTUDIO ENGINE")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def generate_pdf_report(df: pd.DataFrame, session_info: Dict[str, Any], profile_data: Optional[Dict[str, Any]] = None) -> bytes:
    """Generate professional PDF Data Profile Report using ReportLab."""
    if profile_data is None:
        profile_data = generate_full_profile(df, session_info)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#475569")
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    story = []
    overview = profile_data["overview"]

    # ─────────────────────────────────────────────────────────────
    # PAGE 1: COVER PAGE
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 100))
    story.append(Paragraph("DATASTUDIO AI", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#3B82F6"), spaceAfter=8)))
    story.append(Paragraph("Intelligent Data Profile Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Comprehensive Audit & Statistical Architecture for <b>{overview['file_name']}</b>", subtitle_style))
    story.append(Spacer(1, 40))

    # Quality Score Badge Table
    grade = overview["grade"]
    grade_color = "#10B981" if grade in ["A", "B"] else ("#F59E0B" if grade == "C" else "#EF4444")
    score_p = Paragraph(
        f"<font size='36'><b>{overview['quality_score']} / 100</b></font><br/><font size='16'>Grade {grade}</font>",
        ParagraphStyle("ScoreBox", fontName="Helvetica-Bold", alignment=1, textColor=colors.HexColor(grade_color))
    )
    score_table = Table([[score_p]], colWidths=[240], rowHeights=[70])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor(grade_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 40))

    meta_text = (
        f"<b>Dataset Name:</b> {overview['file_name']}<br/>"
        f"<b>Generated On:</b> {datetime.now().strftime('%B %d, %Y - %H:%M UTC')}<br/>"
        f"<b>Dimensions:</b> {overview['total_rows']:,} rows × {overview['total_columns']} columns ({overview['total_cells']:,} cells)<br/>"
        f"<b>Memory Consumption:</b> {overview['memory_mb']} MB"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # PAGE 2: EXECUTIVE SUMMARY & METRICS
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph("Key quality dimensions and global health indicators across the dataset.", body_style))
    story.append(Spacer(1, 10))

    metrics_data = [
        ["Metric", "Value", "Health Status"],
        ["Total Records", f"{overview['total_rows']:,}", "Verified"],
        ["Total Features", str(overview["total_columns"]), "Verified"],
        ["Duplicate Rows", f"{overview['duplicate_rows']} ({overview['duplicate_pct']}%)", "Optimal" if overview['duplicate_rows'] == 0 else "Action Required"],
        ["Missing Cells", f"{overview['missing_cells']} ({overview['missing_pct']}%)", "Clean" if overview['missing_cells'] == 0 else "Incomplete"],
        ["Outlier Rows", f"{profile_data['outlier_analysis']['overall_outlier_rows']}", "Normal" if profile_data['outlier_analysis']['overall_outlier_rows'] == 0 else "Review Needed"],
        ["Quality Score", f"{overview['quality_score']} (Grade {overview['grade']})", "Compliant" if overview['quality_score'] >= 80 else "Needs Cleaning"]
    ]
    t_exec = Table(metrics_data, colWidths=[180, 160, 164])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_exec)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Identified Quality Issues", h2_style))
    issues = profile_data.get("quality_issues", [])
    if issues:
        iss_data = [["Severity", "Identified Anomaly", "Recommended Remediation"]]
        for iss in issues:
            sev = iss.get("severity", "warning").upper()
            iss_data.append([sev, iss.get("issue", ""), iss.get("recommended_fix", "")])
        t_iss = Table(iss_data, colWidths=[80, 220, 204])
        t_iss.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FEF2F2"), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_iss)
    else:
        story.append(Paragraph("No critical quality anomalies were detected.", body_style))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # PAGE 3+: COLUMN PROFILES
    # ─────────────────────────────────────────────────────────────
    col_profs = profile_data.get("column_profiles", {})
    for col_name, prof in col_profs.items():
        story.append(Spacer(1, 15))
        kind = prof.get("kind", "categorical").upper()
        story.append(Paragraph(f"Column Profile: <b>{col_name}</b> <font size='10' color='#64748B'>({kind} • {prof.get('dtype')})</font>", h1_style))

        # Basic Stats Table
        basic_data = [
            ["Attribute", "Value", "Attribute", "Value"],
            ["Total Non-Null", str(prof.get("count")), "Missing Count (%)", f"{prof.get('missing')} ({prof.get('missing_pct')}%)"],
            ["Unique Values", str(prof.get("unique")), "Unique Ratio", f"{prof.get('unique_pct')}%"]
        ]

        if kind == "NUMERIC":
            basic_data.extend([
                ["Mean", str(prof.get("mean")), "Std Deviation", str(prof.get("std"))],
                ["Median", str(prof.get("median")), "IQR", str(prof.get("iqr"))],
                ["Min", str(prof.get("min")), "Max", str(prof.get("max"))],
                ["Skewness", str(prof.get("skewness")), "Kurtosis", str(prof.get("kurtosis"))],
                ["Outliers (IQR)", f"{prof.get('outlier_count')} ({prof.get('outlier_percentage')}%)", "Distribution Type", prof.get("distribution_type", "")]
            ])
        elif kind == "DATE":
            basic_data.extend([
                ["Min Date", str(prof.get("min_date")), "Max Date", str(prof.get("max_date"))],
                ["Date Range (Days)", str(prof.get("date_range_days")), "Gaps Detected", str(prof.get("date_gaps_detected"))]
            ])
        else:
            basic_data.extend([
                ["Avg String Length", str(prof.get("avg_length")), "Min/Max Length", f"{prof.get('min_length')} / {prof.get('max_length')}"],
                ["Contains Digits", str(prof.get("contains_numbers")), "Contains Special", str(prof.get("contains_special_chars"))]
            ])

        t_col = Table(basic_data, colWidths=[120, 132, 120, 132])
        t_col.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F0F9FF"), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_col)

        # Histogram or Top Categories
        if kind == "NUMERIC" and prof.get("histogram"):
            story.append(Spacer(1, 10))
            story.append(Paragraph("10-Bin Distribution Histogram", h2_style))
            hist_table_data = [["Bin Start", "Bin End", "Frequency Count"]]
            for h in prof["histogram"][:10]:
                hist_table_data.append([str(h["bin_start"]), str(h["bin_end"]), str(h["count"])])
            t_hist = Table(hist_table_data, colWidths=[160, 160, 184])
            t_hist.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0F2FE")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(t_hist)

        elif kind == "CATEGORICAL" and prof.get("top_values"):
            story.append(Spacer(1, 10))
            story.append(Paragraph("Top 10 Category Frequencies", h2_style))
            cat_table_data = [["Category Value", "Occurrences"]]
            for cv in prof["top_values"]:
                cat_table_data.append([str(cv["value"])[:40], str(cv["count"])])
            t_cat = Table(cat_table_data, colWidths=[340, 164])
            t_cat.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(t_cat)

        story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # LAST PAGE: RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph("Actionable Recommendations & Next Steps", h1_style))
    story.append(Paragraph("Priority-sorted recommendations to enhance data quality and ML readiness.", body_style))
    story.append(Spacer(1, 12))

    recs = profile_data.get("recommendations", [])[:10]
    if recs:
        rec_table_data = [["Priority", "Category", "Title", "Recommended Action"]]
        for r in recs:
            rec_table_data.append([
                str(r.get("priority", "")),
                r.get("category", ""),
                r.get("title", "")[:40],
                r.get("description", "")[:55] + "..."
            ])
        t_rec = Table(rec_table_data, colWidths=[50, 70, 170, 214])
        t_rec.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_rec)

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
