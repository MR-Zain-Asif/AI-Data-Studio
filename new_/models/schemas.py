"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, Any, Union


class CleanRequest(BaseModel):
    """Request body for cleaning operations."""
    operation: str = Field(..., description="The cleaning operation to perform")
    column: Optional[str] = Field(None, description="Target column name")
    method: Optional[str] = Field(None, description="Method to use (mean/median/mode/custom_value)")
    value: Optional[str] = Field(None, description="Custom value for fill operations")
    new_name: Optional[str] = Field(None, description="New name for rename operations")
    dtype: Optional[str] = Field(None, description="Target data type for conversion")
    date_format: Optional[str] = Field(None, description="Target date format")
    case: Optional[str] = Field(None, description="Case type: upper/lower/title")


class AssistantRequest(BaseModel):
    """Request body for natural language assistant."""
    command: str = Field(..., description="Natural language command")


class ColumnIssue(BaseModel):
    """A single issue found in data analysis."""
    issue_type: str
    severity: str  # critical, warning, info
    column: Optional[str] = None
    description: str
    affected_count: int = 0
    recommendation: str = ""
    operation: Optional[str] = None
    method: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response from the analysis endpoint."""
    quality_score: float
    total_issues: int
    issues: list[ColumnIssue]
    recommendations: list[str]
    summary: dict[str, Any] = {}


class ColumnStats(BaseModel):
    """Statistics for a single column."""
    column_name: str
    data_type: str
    total_count: int
    missing_count: int
    missing_percentage: float
    unique_count: Optional[int] = None
    outlier_count: int = 0
    # Numeric stats
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_value: Optional[float] = None
    # Text stats
    most_common: Optional[str] = None
    shortest_value: Optional[str] = None
    longest_value: Optional[str] = None


class StatsResponse(BaseModel):
    """Response from the statistics endpoint."""
    columns: list[ColumnStats]
    row_count: int
    column_count: int


class UploadResponse(BaseModel):
    """Response from the upload endpoint."""
    session_id: str
    row_count: int
    column_count: int
    column_names: list[str]
    file_size: int
    file_name: str
    preview: list[dict[str, Any]]


class CleanResponse(BaseModel):
    """Response from a clean operation."""
    success: bool
    message: str
    rows_affected: int = 0
    new_row_count: int = 0
    new_column_count: int = 0
    quality_score: float = 0.0
    preview: list[dict[str, Any]] = []


class AssistantResponse(BaseModel):
    """Response from the natural language assistant."""
    success: bool
    interpreted_as: str
    message: str
    operation_result: Optional[dict[str, Any]] = None
    suggestion: Optional[str] = None
    confidence: Optional[int] = None


class HistoryEntry(BaseModel):
    """A single entry in the operation history."""
    timestamp: str
    operation: str
    description: str
    rows_affected: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"


# ── Feature 1: Pipeline Schemas ───────────────────────────────────
class PipelineSaveRequest(BaseModel):
    session_id: Optional[str] = None
    name: str = Field(..., description="Pipeline name")
    steps: list[dict[str, Any]] = Field(..., description="List of pipeline steps")


class PipelineRunRequest(BaseModel):
    pipeline_name: str = Field(..., description="Name of pipeline to run")


# ── Feature 3: Data Validation Schemas ────────────────────────────
class RuleObject(BaseModel):
    column: str
    rule_type: str
    value: Optional[Any] = None
    friendly_name: Optional[str] = None


class ValidationRequest(BaseModel):
    rules: list[RuleObject]


# ── Feature 4: Privacy Schemas ────────────────────────────────────
class MaskPIIRequest(BaseModel):
    column: str
    pii_type: str


# ── Feature 5: Two File Comparison Schemas ───────────────────────
class CompareRequest(BaseModel):
    session_id_1: str
    session_id_2: str
    name1: Optional[str] = "Dataset 1"
    name2: Optional[str] = "Dataset 2"


# ── Feature 7: Excel Sheet Selection Schema ───────────────────────
class SelectSheetRequest(BaseModel):
    sheet_name: str


# ── Feature 9: Column Merge Schema ────────────────────────────────
class MergeColumnsRequest(BaseModel):
    col1: str
    col2: str
    strategy: str  # prefer_first, prefer_second, concatenate, sum, average
    new_name: str
