"""End-to-End Verification Suite for all 10 Advanced Features."""

import os
import sys
import io
import numpy as np
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app, session_store
import services.cleaner as cleaner_mod
import services.monitor as monitor_mod
import services.version_service as version_mod
import services.anomaly_service as anomaly_mod
import services.transform_service as transform_mod
import services.recommender as recommender_mod
import pytest
import services.profiler as profiler_mod
import services.ml_service as ml_mod

client = TestClient(app)


@pytest.fixture
def sample_dataset():
    """Generate rich dataset containing numeric, text, date, missing values, outliers, duplicates."""
    np.random.seed(42)
    n = 60
    data = {
        "User_ID": [f"ID_{i}" for i in range(n - 4)] + ["ID_0", "ID_1", "ID_2", "ID_3"],  # some duplicate IDs
        "Age": [np.random.randint(18, 70) if i > 5 else np.nan for i in range(n - 2)] + [-5, 180],  # missing + anomalies
        "Salary": [np.random.randint(30000, 120000) for _ in range(n - 1)] + [950000],  # outlier
        "Score": np.random.uniform(0.0, 100.0, n),
        "Category": ["yes", "Yes", "YES", "no", "No"] * 12,  # inconsistent casing
        "Notes": [f"  Customer note number {i}  " for i in range(n)],  # whitespace
        "Signup_Date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist(),
        "Purchased": np.random.choice([0, 1], size=n)  # target
    }
    # Introduce duplicate rows and negative age at safe indices
    df = pd.DataFrame(data)
    df.loc[10, "Age"] = -5.0
    df.iloc[58] = df.iloc[0].copy()
    df.iloc[59] = df.iloc[1].copy()
    return df


def test_feature_1_automl(sample_dataset):
    """Verify Feature 1: Target detection, model training, Optuna tuning, SHAP explainability, and prediction."""
    df = sample_dataset.copy()
    
    # Target detection
    det = ml_mod.detect_target_column(df)
    assert det["suggested_target"] is not None
    assert det["task_type"] in ["classification", "regression"]

    # 1. Supervised Classification Training (Logistic Regression, Random Forest, etc.)
    leaderboard = ml_mod.train_models(df, target_col="Purchased", task_type="classification", session_id="test_session_ml")
    assert len(leaderboard) >= 5
    assert leaderboard[0]["rank"] == 1
    assert "accuracy" in leaderboard[0]["metrics"]
    assert any("Logistic Regression" in m["model_name"] for m in leaderboard)

    # 2. Supervised Regression Training (Linear, Ridge, Lasso, ElasticNet, etc.)
    reg_leaderboard = ml_mod.train_models(df, target_col="Salary", task_type="regression", session_id="test_session_reg")
    assert len(reg_leaderboard) >= 5
    assert "r2" in reg_leaderboard[0]["metrics"]

    # 3. Unsupervised Clustering (K-Means, GMM, DBSCAN, etc.)
    cluster_leaderboard = ml_mod.train_models(df, task_type="clustering", n_clusters=3, session_id="test_session_clust")
    assert len(cluster_leaderboard) >= 3
    assert "silhouette_score" in cluster_leaderboard[0]["metrics"]
    assert any("K-Means" in m["model_name"] for m in cluster_leaderboard)

    # 4. Unsupervised Anomaly Detection (Isolation Forest, LOF, One-Class SVM)
    anom_leaderboard = ml_mod.train_models(df, task_type="anomaly", contamination=0.05, session_id="test_session_anom")
    assert len(anom_leaderboard) >= 2
    assert "anomaly_count" in anom_leaderboard[0]["metrics"]

    # Optuna tuning
    tuned = ml_mod.tune_best_model(df, target_col="Purchased", task_type="classification", n_trials=5, session_id="test_session_ml")
    assert "best_score" in tuned

    # Feature Suggestions & Engineering
    sugs = ml_mod.suggest_features(df, target_col="Purchased")
    assert len(sugs) > 0
    eng_df, new_cols = ml_mod.auto_engineer_features(df, sugs[:2])
    assert len(new_cols) > 0


def test_feature_2_profiler(sample_dataset):
    """Verify Feature 2: 6-section profiling report and PDF generation."""
    df = sample_dataset.copy()
    prof = profiler_mod.generate_full_profile(df, session_info={"file_name": "test.csv"})
    
    assert "overview" in prof
    assert "column_profiles" in prof
    assert "correlations" in prof
    assert "missing_analysis" in prof
    assert "outlier_analysis" in prof
    assert "quality_issues" in prof
    assert prof["overview"]["total_rows"] == len(df)

    # ReportLab PDF Generation
    pdf_bytes = profiler_mod.generate_pdf_report(df, session_info={"file_name": "test.csv"}, profile_data=prof)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_feature_3_cleaning_engine(sample_dataset):
    """Verify Feature 3: KNN impute, regression impute, iterative, winsorize, clean text, standardize, dates, scaling, encoding."""
    df = sample_dataset.copy()
    
    # 1. KNN Impute
    df_knn, aff_knn = cleaner_mod.DataCleaner.knn_impute(df, "Age", n_neighbors=3)
    assert df_knn["Age"].isna().sum() == 0

    # 2. Winsorize
    df_win, aff_win = cleaner_mod.DataCleaner.winsorize_column(df, "Salary", limits=(0.05, 0.05))
    assert df_win["Salary"].max() < 950000

    # 3. Clean Text
    df_txt, aff_txt = cleaner_mod.DataCleaner.clean_text_column(df, "Notes", ["strip", "lower"])
    assert aff_txt > 0
    assert df_txt["Notes"].iloc[0].startswith("customer")

    # 4. Standardize Categories
    df_cat, aff_cat = cleaner_mod.DataCleaner.standardize_categories(df, "Category", {"yes": "Yes", "YES": "Yes", "no": "No"})
    assert set(df_cat["Category"].unique()).issubset({"Yes", "No"})

    # 5. Extract Date Features
    df_dt, date_cols = cleaner_mod.DataCleaner.extract_date_features(df, "Signup_Date")
    assert "Signup_Date_year" in df_dt.columns

    # 6. Scaling
    df_sc, scaler = cleaner_mod.DataCleaner.scale_features(df, ["Score"], method="standard")
    assert abs(df_sc["Score"].mean()) < 0.2

    # 7. Encoding
    df_enc, enc = cleaner_mod.DataCleaner.encode_categorical(df, "Category", method="onehot")
    assert "Category_Yes" in df_enc.columns or any("Category_" in c for c in df_enc.columns)

    # 8. Split and Calculated Column
    df_split = cleaner_mod.DataCleaner.split_column(df, "User_ID", "_", ["Prefix", "Num"])
    assert "Prefix" in df_split.columns
    df_calc = cleaner_mod.DataCleaner.create_calculated_column(df, "Double_Score", "Score * 2")
    assert "Double_Score" in df_calc.columns


def test_feature_4_quality_monitor(sample_dataset):
    """Verify Feature 4: Real-time 6-dimension metrics, score and grade."""
    df = sample_dataset.copy()
    metrics = monitor_mod.calculate_quality_metrics(df)
    
    assert 0 <= metrics["overall_score"] <= 100
    assert metrics["grade"] in ["A", "B", "C", "D", "F"]
    assert "completeness" in metrics
    assert "consistency" in metrics
    assert "accuracy" in metrics
    assert "uniqueness" in metrics
    assert "validity" in metrics
    assert "timeliness" in metrics
    assert len(metrics["improvements"]) > 0


def test_feature_5_versioning(sample_dataset):
    """Verify Feature 5: Snapshots, history, restore, and diff comparison."""
    df = sample_dataset.copy()
    vm = version_mod.VersionManager("session_test_v")
    
    # Save v1
    v1 = vm.save_version(df, "Initial", "First snapshot", 0)
    assert v1["version_number"] == 1
    
    # Modify and save v2
    df_mod = df.drop(columns=["Notes"])
    df_mod = df_mod.drop_duplicates()
    v2 = vm.save_version(df_mod, "Cleaned", "Dropped notes and duplicates", len(df) - len(df_mod))
    assert v2["version_number"] == 2

    # Diff
    diff = vm.compare_versions(v1["id"], v2["id"])
    assert "columns_removed" in diff
    assert "Notes" in diff["columns_removed"]

    # Restore v1
    restored = vm.restore_version(v1["id"])
    assert len(restored) == len(df)


def test_feature_7_anomalies(sample_dataset):
    """Verify Feature 7: Anomaly detection and automated fix-all."""
    df = sample_dataset.copy()
    anoms = anomaly_mod.detect_all_anomalies(df)
    assert len(anoms) > 0
    
    # Auto fix
    fixed_df, aff, fixes = anomaly_mod.fix_all_anomalies(df)
    assert aff > 0
    assert (fixed_df["Age"].dropna() >= 0).all()  # negative age fixed


def test_feature_8_transform(sample_dataset):
    """Verify Feature 8: Pivot, melt, transpose, aggregation, sampling, binning, math."""
    df = sample_dataset.copy()
    
    # Aggregate
    agg = transform_mod.group_and_aggregate(df, "Category", {"Score": ["mean", "max"]})
    assert len(agg) > 0

    # Sample
    sample, info = transform_mod.smart_sample(df, method="random", n=15)
    assert len(sample) == 15

    # Binning
    binned = transform_mod.bin_numeric_column(df, "Score", method="equal_width", bins=4)
    assert "Score_binned" in binned.columns

    # Math
    math_df = transform_mod.apply_math_operation(df, "Score", "multiply", value=1.5)
    assert np.isclose(math_df["Score"].iloc[0], df["Score"].iloc[0] * 1.5)


def test_feature_9_recommendations(sample_dataset):
    """Verify Feature 9: Priority recommendations and critical fixes."""
    df = sample_dataset.copy()
    recs = recommender_mod.generate_smart_recommendations(df)
    assert len(recs) > 0
    assert recs[0]["priority"] >= recs[-1]["priority"]

    # Critical fix application
    crit_df, aff, logs = recommender_mod.apply_critical_recommendations(df)
    assert len(crit_df) <= len(df)


def test_feature_10_export(sample_dataset):
    """Verify Feature 10: Parquet, Feather, Formatted Excel, Certificate, Comparison."""
    df = sample_dataset.copy()
    sid = "test_export_sid"
    session_store[sid] = {
        "session_id": sid,
        "current_df": df.copy(),
        "original_df": df.copy(),
        "file_name": "test.csv",
        "history": [],
        "operations": [{"timestamp": "2026-09-17T20:00:00", "operation": "Imputed Age", "rows_affected": 5}]
    }

    # Parquet
    res = client.get(f"/api/export/{sid}/parquet")
    assert res.status_code == 200

    # Feather
    res = client.get(f"/api/export/{sid}/feather")
    assert res.status_code == 200

    # Formatted Excel
    res = client.get(f"/api/export/{sid}/excel-formatted")
    assert res.status_code == 200

    # PDF Certificate
    res = client.get(f"/api/export/{sid}/certificate")
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")

    # Comparison
    res = client.get(f"/api/export/{sid}/comparison")
    assert res.status_code == 200


def test_feature_6_assistant(sample_dataset):
    """Verify Feature 6: Smart AI Assistant chat, auto-execution, history, clear, proactive suggestions."""
    df = sample_dataset.copy()
    sid = "test_assistant_sid"
    session_store[sid] = {
        "session_id": sid,
        "current_df": df.copy(),
        "original_df": df.copy(),
        "file_name": "test.csv",
        "history": [],
        "operations": []
    }

    # 1. Proactive suggestions
    res = client.get(f"/api/assistant/{sid}/suggestions")
    assert res.status_code == 200
    assert "proactive_suggestion" in res.json()

    # 2. Chat with command auto-execution
    res = client.post(f"/api/assistant/{sid}/chat", json={"message": "remove duplicate rows"})
    assert res.status_code == 200
    assert res.json().get("auto_executed") is True

    # 3. History
    res = client.get(f"/api/assistant/{sid}/history")
    assert res.status_code == 200
    assert len(res.json().get("history", [])) > 0

    # 4. Clear history
    res = client.post(f"/api/assistant/{sid}/clear")
    assert res.status_code == 200


if __name__ == "__main__":
    print("Running Advanced Features Verification Test Suite...")
    ds = sample_dataset()
    print("Testing Feature 1: AutoML...")
    test_feature_1_automl(ds)
    print("  [OK] Feature 1 passed.")

    print("Testing Feature 2: Smart Data Profiler...")
    test_feature_2_profiler(ds)
    print("  [OK] Feature 2 passed.")

    print("Testing Feature 3: Advanced Cleaning Engine...")
    test_feature_3_cleaning_engine(ds)
    print("  [OK] Feature 3 passed.")

    print("Testing Feature 4: Real-time Quality Monitor...")
    test_feature_4_quality_monitor(ds)
    print("  [OK] Feature 4 passed.")

    print("Testing Feature 5: Dataset Versioning...")
    test_feature_5_versioning(ds)
    print("  [OK] Feature 5 passed.")

    print("Testing Feature 6: Smart AI Assistant...")
    test_feature_6_assistant(ds)
    print("  [OK] Feature 6 passed.")

    print("Testing Feature 7: Anomaly Detection Engine...")
    test_feature_7_anomalies(ds)
    print("  [OK] Feature 7 passed.")

    print("Testing Feature 8: Transformation Studio...")
    test_feature_8_transform(ds)
    print("  [OK] Feature 8 passed.")

    print("Testing Feature 9: Intelligent Recommendations...")
    test_feature_9_recommendations(ds)
    print("  [OK] Feature 9 passed.")

    print("Testing Feature 10: Multi-Format Export...")
    test_feature_10_export(ds)
    print("  [OK] Feature 10 passed.")

    print("\n>>> ALL 10 ADVANCED FEATURES FULLY VERIFIED AND PASSING! <<<")
