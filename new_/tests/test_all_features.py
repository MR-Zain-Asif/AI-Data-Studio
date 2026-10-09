import os
import sys
import pytest
import io
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app, session_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    session_store.clear()


def create_sample_csv():
    df = pd.DataFrame({
        "First Name": [" John ", "Jane ", "John ", " Alice"],
        "FirstName": ["John", "Jane", "John", "Alice"],
        "Age": [25, 150, 25, 30],
        "Email": ["john@example.com", "invalid-email", "john@example.com", "alice@example.com"],
        "Phone": ["123-456-7890", "9876543210", "123-456-7890", "555-123-4567"],
        "Salary": [50000, 60000, 50000, 70000]
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def upload_test_session():
    csv_buf = create_sample_csv()
    response = client.post(
        "/api/upload",
        files={"file": ("test_dataset.csv", csv_buf, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    return data["session_id"]


def test_table_previews():
    sid = upload_test_session()

    # Current preview table
    res1 = client.get(f"/api/preview/{sid}?page=1&page_size=50")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total_rows"] == 4
    assert len(data1["rows"]) == 4
    assert len(data1["columns"]) == 6

    # Original preview table
    res2 = client.get(f"/api/original-preview/{sid}?page=1&page_size=50")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["total_rows"] == 4
    assert len(data2["rows"]) == 4


def test_feature1_pipeline_system():
    sid = upload_test_session()

    # List pipelines (presets + user)
    res = client.get("/api/pipeline/list")
    assert res.status_code == 200
    pdata = res.json()
    assert "Sales Data" in pdata["presets"]

    # Run preset pipeline
    res_run = client.post(
        f"/api/pipeline/run/{sid}",
        json={"pipeline_name": "Sales Data"}
    )
    assert res_run.status_code == 200
    assert "Sales Data" in res_run.json()["message"]

    # Save custom pipeline
    res_save = client.post(
        "/api/pipeline/save",
        json={"session_id": sid, "name": "Custom Test Pipeline", "steps": []}
    )
    assert res_save.status_code == 200
    assert "saved successfully" in res_save.json()["message"]


def test_feature2_pdf_report():
    sid = upload_test_session()
    res = client.get(f"/api/export/{sid}/report/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 100  # PDF content generated successfully


def test_feature3_data_validation():
    sid = upload_test_session()

    # Get rule templates
    tpl_res = client.get("/api/validate/templates")
    assert tpl_res.status_code == 200
    assert len(tpl_res.json()["templates"]) > 0

    # Run validation rules
    rules = [
        {"column": "Age", "rule_type": "min_value", "value": 0, "friendly_name": "Min Age"},
        {"column": "Age", "rule_type": "max_value", "value": 120, "friendly_name": "Max Age"},
        {"column": "Email", "rule_type": "email_format", "value": None, "friendly_name": "Email Format"}
    ]
    val_res = client.post(f"/api/validate/{sid}", json={"rules": rules})
    assert val_res.status_code == 200
    vdata = val_res.json()
    assert vdata["total_rules"] == 3
    assert vdata["failed_rules"] >= 1  # Age 150 & invalid email should fail


def test_feature4_pii_privacy_scan_and_mask():
    sid = upload_test_session()

    # Scan PII
    scan_res = client.get(f"/api/privacy/scan/{sid}")
    assert scan_res.status_code == 200
    sdata = scan_res.json()
    assert sdata["total_pii_found"] >= 1

    # Mask PII
    mask_res = client.post(
        f"/api/privacy/mask/{sid}",
        json={"column": "Email", "pii_type": "Email Address"}
    )
    assert mask_res.status_code == 200
    assert mask_res.json()["rows_affected"] == 4


def test_feature5_dataset_comparison():
    sid1 = upload_test_session()
    sid2 = upload_test_session()

    comp_res = client.post(
        "/api/compare",
        json={"session_id_1": sid1, "session_id_2": sid2, "name1": "Set A", "name2": "Set B"}
    )
    assert comp_res.status_code == 200
    cdata = comp_res.json()
    assert cdata["dataset_1"]["rows"] == 4
    assert cdata["dataset_2"]["rows"] == 4


def test_feature6_column_relationships():
    sid = upload_test_session()
    rel_res = client.get(f"/api/relationships/{sid}")
    assert rel_res.status_code == 200
    rdata = rel_res.json()
    assert "correlation_matrix" in rdata
    assert "duplicate_column_pairs" in rdata


def test_feature8_google_sheets_export():
    sid = upload_test_session()

    # UTF-8 BOM CSV export
    csv_res = client.get(f"/api/export/{sid}/sheets-csv")
    assert csv_res.status_code == 200
    assert b"\xef\xbb\xbf" in csv_res.content  # UTF-8 BOM byte marker present

    # TSV Clipboard export
    tsv_res = client.post(f"/api/export/{sid}/sheets-url")
    assert tsv_res.status_code == 200
    assert "tsv_data" in tsv_res.json()


def test_feature9_similar_columns_and_merge():
    sid = upload_test_session()

    # Find similar columns ("First Name" vs "FirstName")
    sim_res = client.get(f"/api/columns/similar/{sid}")
    assert sim_res.status_code == 200
    pairs = sim_res.json()["similar_pairs"]
    assert len(pairs) >= 1
    assert pairs[0]["col1"] in ["First Name", "FirstName"]

    # Merge columns
    merge_res = client.post(
        f"/api/columns/merge/{sid}",
        json={
            "col1": "First Name",
            "col2": "FirstName",
            "strategy": "prefer_first",
            "new_name": "Full_Name"
        }
    )
    assert merge_res.status_code == 200
    assert "Merged" in merge_res.json()["message"]


def test_feature10_groq_llm_assistant():
    sid = upload_test_session()

    # Test auto clean command via Groq LLM Assistant
    res1 = client.post(f"/api/assistant/{sid}", json={"command": "auto clean dataset"})
    assert res1.status_code == 200
    assert res1.json()["success"] is True
    assert len(res1.json()["result_message"]) > 10

    # Test privacy scan command via Groq LLM Assistant
    res2 = client.post(f"/api/assistant/{sid}", json={"command": "explain dataset cleaning steps done so far"})
    assert res2.status_code == 200
    assert res2.json()["success"] is True
    assert "groq" in res2.json()["interpreted_as"].lower()
