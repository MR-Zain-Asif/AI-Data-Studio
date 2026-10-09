"""Dataset Versioning System Service."""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from services.monitor import calculate_quality_metrics


class VersionManager:
    """Manages snapshot versions of datasets for a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.versions: List[Dict[str, Any]] = []
        self.current_index: int = -1

    def save_version(
        self,
        df: pd.DataFrame,
        operation_name: str,
        description: str,
        rows_affected: int = 0
    ) -> Dict[str, Any]:
        """Save a new version snapshot."""
        try:
            quality = calculate_quality_metrics(df)
            score = float(quality.get("overall_score", 0.0))
        except Exception:
            score = 0.0

        v_num = len(self.versions) + 1
        v_id = str(uuid.uuid4())
        version_obj = {
            "id": v_id,
            "version_number": v_num,
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "description": description,
            "rows_affected": int(rows_affected),
            "quality_score": score,
            "df_snapshot": df.copy(),
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns": [str(c) for c in df.columns]
        }
        self.versions.append(version_obj)
        self.current_index = len(self.versions) - 1

        # Return metadata without df_snapshot
        return self._meta_only(version_obj)

    def _meta_only(self, v: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": v["id"],
            "version_number": v["version_number"],
            "timestamp": v["timestamp"],
            "operation": v["operation"],
            "description": v["description"],
            "rows_affected": v["rows_affected"],
            "quality_score": v["quality_score"],
            "shape": v["shape"],
            "columns": v["columns"],
            "is_current": (self.current_index >= 0 and self.versions[self.current_index]["id"] == v["id"])
        }

    def get_version(self, version_id: str) -> Optional[pd.DataFrame]:
        for v in self.versions:
            if v["id"] == version_id:
                return v["df_snapshot"].copy()
        return None

    def get_version_meta(self, version_id: str) -> Optional[Dict[str, Any]]:
        for v in self.versions:
            if v["id"] == version_id:
                return self._meta_only(v)
        return None

    def restore_version(self, version_id: str) -> Optional[pd.DataFrame]:
        for idx, v in enumerate(self.versions):
            if v["id"] == version_id:
                self.current_index = idx
                return v["df_snapshot"].copy()
        return None

    def compare_versions(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """Compare two version snapshots and generate diff report."""
        v1_obj = next((v for v in self.versions if v["id"] == v1_id), None)
        v2_obj = next((v for v in self.versions if v["id"] == v2_id), None)

        if not v1_obj or not v2_obj:
            raise ValueError(f"One or both versions ({v1_id}, {v2_id}) not found")

        df1 = v1_obj["df_snapshot"]
        df2 = v2_obj["df_snapshot"]

        rows1, cols1 = len(df1), set(df1.columns)
        rows2, cols2 = len(df2), set(df2.columns)

        rows_added = max(0, rows2 - rows1)
        rows_removed = max(0, rows1 - rows2)

        columns_added = [str(c) for c in cols2 if c not in cols1]
        columns_removed = [str(c) for c in cols1 if c not in cols2]

        # Calculate cells changed on shared row indices and columns
        common_cols = list(cols1.intersection(cols2))
        cells_changed = 0
        min_rows = min(rows1, rows2)
        if common_cols and min_rows > 0:
            try:
                sub1 = df1.iloc[:min_rows][common_cols]
                sub2 = df2.iloc[:min_rows][common_cols]
                # Compare handling NaNs
                diff_mask = (sub1.values != sub2.values) & ~(pd.isna(sub1.values) & pd.isna(sub2.values))
                cells_changed = int(diff_mask.sum())
            except Exception:
                cells_changed = 0

        cells_changed += abs(rows2 - rows1) * len(common_cols)

        quality_change = round(v2_obj["quality_score"] - v1_obj["quality_score"], 2)

        return {
            "v1_id": v1_id,
            "v2_id": v2_id,
            "v1_number": v1_obj["version_number"],
            "v2_number": v2_obj["version_number"],
            "rows_added": int(rows_added),
            "rows_removed": int(rows_removed),
            "columns_added": columns_added,
            "columns_removed": columns_removed,
            "cells_changed": int(cells_changed),
            "quality_change": float(quality_change),
            "v1_quality": float(v1_obj["quality_score"]),
            "v2_quality": float(v2_obj["quality_score"]),
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return [self._meta_only(v) for v in self.versions]
