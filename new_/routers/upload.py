"""Upload router — handles file upload, multi-sheet excel parsing, table pagination previews, and session creation."""

import uuid
import io
import math
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from models.schemas import SelectSheetRequest

router = APIRouter()

_session_store: dict = None


def set_session_store(store: dict):
    global _session_store
    _session_store = store


def get_session_store() -> dict:
    global _session_store
    if _session_store is None:
        _session_store = {}
    return _session_store


def _get_session(session_id: str) -> dict:
    store = get_session_store()
    if session_id not in store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return store[session_id]


def _sanitize_val(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV, Excel, or JSON file for analysis and cleaning."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        filename = file.filename.lower()
        allowed_extensions = (".csv", ".xlsx", ".xls", ".json")
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
            )

        content = await file.read()
        file_size = len(content)
        sheet_names = []
        is_multi_sheet = False

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
                excel_file = pd.ExcelFile(io.BytesIO(content), engine=engine)
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    is_multi_sheet = True
                df = pd.read_excel(excel_file, sheet_name=sheet_names[0])
            elif filename.endswith(".json"):
                df = pd.read_json(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or has no data")

        session_id = str(uuid.uuid4())[:12]
        store = get_session_store()
        store[session_id] = {
            "session_id": session_id,
            "original_df": df.copy(),
            "current_df": df.copy(),
            "history": [],
            "redo_stack": [],
            "operations": [],
            "file_name": file.filename,
            "raw_content": content,
            "sheet_names": sheet_names,
            "active_sheet": sheet_names[0] if sheet_names else None,
        }

        preview_df = df.head(10).copy()
        raw_rows = preview_df.to_dict(orient="records")
        preview = [{k: _sanitize_val(val) for k, val in row.items()} for row in raw_rows]

        return JSONResponse(content={
            "session_id": session_id,
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_names": df.columns.tolist(),
            "file_size": file_size,
            "file_name": file.filename,
            "is_multi_sheet": is_multi_sheet,
            "sheet_names": sheet_names,
            "active_sheet": sheet_names[0] if sheet_names else None,
            "preview": preview,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/preview/{session_id}")
async def get_table_preview(session_id: str, page: int = 1, page_size: int = 50):
    """Return paginated preview records for current dataframe state."""
    try:
        session = _get_session(session_id)
        df = session["current_df"]
        total_rows = len(df)
        total_pages = max(1, math.ceil(total_rows / page_size))

        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        sliced_df = df.iloc[start_idx:end_idx].copy()
        raw_rows = sliced_df.to_dict(orient="records")
        rows = [{k: _sanitize_val(val) for k, val in row.items()} for row in raw_rows]

        return JSONResponse(content={
            "columns": df.columns.tolist(),
            "rows": rows,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch table records: {str(e)}")


@router.get("/original-preview/{session_id}")
async def get_original_table_preview(session_id: str, page: int = 1, page_size: int = 50):
    """Return paginated preview records for original initial dataframe state."""
    try:
        session = _get_session(session_id)
        df = session.get("original_df", session["current_df"])
        total_rows = len(df)
        total_pages = max(1, math.ceil(total_rows / page_size))

        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        sliced_df = df.iloc[start_idx:end_idx].copy()
        raw_rows = sliced_df.to_dict(orient="records")
        rows = [{k: _sanitize_val(val) for k, val in row.items()} for row in raw_rows]

        return JSONResponse(content={
            "columns": df.columns.tolist(),
            "rows": rows,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch original table records: {str(e)}")


@router.get("/upload/sheets/{session_id}")
async def get_sheets(session_id: str):
    """Return available Excel sheet names and row counts."""
    try:
        session = _get_session(session_id)
        sheet_names = session.get("sheet_names", [])
        content = session.get("raw_content")
        filename = session.get("file_name", "").lower()

        if not sheet_names or not content:
            return JSONResponse(content={"is_multi_sheet": False, "sheets": []})

        engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
        excel_file = pd.ExcelFile(io.BytesIO(content), engine=engine)

        sheets_info = []
        for name in sheet_names:
            sheet_df = pd.read_excel(excel_file, sheet_name=name)
            sheets_info.append({
                "sheet_name": name,
                "row_count": len(sheet_df),
                "column_count": len(sheet_df.columns)
            })

        return JSONResponse(content={
            "is_multi_sheet": len(sheet_names) > 1,
            "active_sheet": session.get("active_sheet"),
            "sheets": sheets_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sheets: {str(e)}")


@router.post("/upload/select-sheet/{session_id}")
async def select_sheet(session_id: str, request: SelectSheetRequest):
    """Load a specific sheet from multi-sheet Excel file."""
    try:
        session = _get_session(session_id)
        content = session.get("raw_content")
        filename = session.get("file_name", "").lower()

        if not content:
            raise HTTPException(status_code=400, detail="File content not found in session")

        engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
        excel_file = pd.ExcelFile(io.BytesIO(content), engine=engine)

        if request.sheet_name not in excel_file.sheet_names:
            raise HTTPException(status_code=400, detail=f"Sheet '{request.sheet_name}' not found")

        df = pd.read_excel(excel_file, sheet_name=request.sheet_name)
        if df.empty:
            raise HTTPException(status_code=400, detail=f"Sheet '{request.sheet_name}' is empty")

        # Reset session state for new sheet
        session["original_df"] = df.copy()
        session["current_df"] = df.copy()
        session["history"] = []
        session["redo_stack"] = []
        session["operations"] = []
        session["active_sheet"] = request.sheet_name
        session.pop("cached_analysis", None)
        session.pop("cached_stats", None)

        preview_df = df.head(10).copy()
        raw_rows = preview_df.to_dict(orient="records")
        preview = [{k: _sanitize_val(val) for k, val in row.items()} for row in raw_rows]

        return JSONResponse(content={
            "success": True,
            "active_sheet": request.sheet_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_names": df.columns.tolist(),
            "preview": preview
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sheet selection failed: {str(e)}")
