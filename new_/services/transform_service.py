"""Data Transformation Studio Service."""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


def pivot_table(
    df: pd.DataFrame,
    index: Union[str, List[str]],
    columns: Union[str, List[str]],
    values: Union[str, List[str]],
    aggfunc: str = "mean"
) -> pd.DataFrame:
    """Create pivot table with specified aggregation function."""
    valid_aggs = ["sum", "mean", "count", "min", "max", "std", "median"]
    if aggfunc.lower() not in valid_aggs:
        aggfunc = "mean"

    pivoted = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc.lower(),
        fill_value=0
    )

    # Flatten multi-level columns if any
    if isinstance(pivoted.columns, pd.MultiIndex):
        pivoted.columns = ["_".join(str(c) for c in col).strip("_") for col in pivoted.columns]

    pivoted = pivoted.reset_index()
    return pivoted


def melt_table(
    df: pd.DataFrame,
    id_vars: Optional[List[str]] = None,
    value_vars: Optional[List[str]] = None,
    var_name: str = "variable",
    value_name: str = "value"
) -> pd.DataFrame:
    """Unpivot a DataFrame from wide to long format."""
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name
    )
    return melted


def transpose(df: pd.DataFrame) -> pd.DataFrame:
    """Flip rows and columns, resetting indices to create a clean DataFrame."""
    transposed = df.transpose()
    transposed.columns = [f"row_{i}" for i in range(len(transposed.columns))]
    transposed = transposed.reset_index().rename(columns={"index": "original_column"})
    return transposed


def group_and_aggregate(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    aggregations: Dict[str, List[str]]
) -> pd.DataFrame:
    """Group by specified columns and compute multiple aggregations."""
    if isinstance(group_by, str):
        group_by = [group_by]

    # Validate columns and agg functions
    cleaned_aggs = {}
    valid_funcs = {"sum", "mean", "count", "min", "max", "std", "median", "nunique"}
    for col, funcs in aggregations.items():
        if col in df.columns:
            cleaned_aggs[col] = [f for f in funcs if f.lower() in valid_funcs]

    if not cleaned_aggs:
        raise ValueError("No valid column aggregations provided")

    grouped = df.groupby(group_by, as_index=False).agg(cleaned_aggs)

    # Flatten multi-level column names
    if isinstance(grouped.columns, pd.MultiIndex):
        flat_cols = []
        for c in grouped.columns:
            if c[1] == "" or c[0] in group_by:
                flat_cols.append(str(c[0]))
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        grouped.columns = flat_cols

    return grouped


def smart_sample(
    df: pd.DataFrame,
    method: str = "random",
    n: Optional[int] = None,
    frac: Optional[float] = None,
    stratify_col: Optional[str] = None
) -> (pd.DataFrame, Dict[str, Any]):
    """Sample data using random, stratified, systematic, head, or tail method."""
    total_rows = len(df)
    if total_rows == 0:
        return df.copy(), {"total_rows": 0, "sample_rows": 0, "method": method}

    if frac is not None:
        n = max(1, int(total_rows * min(1.0, max(0.01, frac))))
    elif n is None:
        n = min(100, total_rows)
    else:
        n = min(max(1, n), total_rows)

    method = method.lower()
    if method == "head":
        sampled = df.head(n).copy()
    elif method == "tail":
        sampled = df.tail(n).copy()
    elif method == "systematic":
        step = max(1, total_rows // n)
        sampled = df.iloc[::step].head(n).copy()
    elif method == "stratified" and stratify_col and stratify_col in df.columns:
        # Grouped proportional sample
        strat_frac = n / total_rows
        sampled = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(max(1, int(round(len(x) * strat_frac))))
        ).reset_index(drop=True)
    else:
        # Random sample
        sampled = df.sample(n=n, random_state=42).reset_index(drop=True)

    info = {
        "original_rows": total_rows,
        "sample_rows": len(sampled),
        "sampling_ratio": round(len(sampled) / max(total_rows, 1), 4),
        "method": method
    }
    return sampled, info


def bin_numeric_column(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",
    bins: Union[int, List[float]] = 5,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """Bin continuous numerical feature into categorical intervals."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in DataFrame")

    new_df = df.copy()
    s = pd.to_numeric(new_df[column], errors="coerce")
    new_col_name = f"{column}_binned"

    if method == "equal_frequency":
        q = int(bins) if isinstance(bins, (int, float)) else 5
        new_df[new_col_name] = pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(str)
    elif method == "custom" and isinstance(bins, list) and len(bins) >= 2:
        new_df[new_col_name] = pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
    else:  # equal_width
        b = int(bins) if isinstance(bins, (int, float)) else 5
        new_df[new_col_name] = pd.cut(s, bins=b, labels=labels, include_lowest=True).astype(str)

    return new_df


def apply_math_operation(
    df: pd.DataFrame,
    column: str,
    operation: str,
    value: Optional[float] = None,
    col2: Optional[str] = None
) -> pd.DataFrame:
    """Apply vectorized arithmetic or mathematical operations on columns."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in DataFrame")

    new_df = df.copy()
    s1 = pd.to_numeric(new_df[column], errors="coerce")
    op = operation.lower()

    # Second operand
    if col2 and col2 in new_df.columns:
        s2 = pd.to_numeric(new_df[col2], errors="coerce")
    elif value is not None:
        s2 = float(value)
    else:
        s2 = None

    if op == "add":
        if s2 is None: raise ValueError("Value or col2 required for addition")
        new_df[column] = s1 + s2
    elif op == "subtract":
        if s2 is None: raise ValueError("Value or col2 required for subtraction")
        new_df[column] = s1 - s2
    elif op == "multiply":
        if s2 is None: raise ValueError("Value or col2 required for multiplication")
        new_df[column] = s1 * s2
    elif op == "divide":
        if s2 is None: raise ValueError("Value or col2 required for division")
        denom = s2 if isinstance(s2, (int, float)) else s2.replace(0, np.nan)
        if isinstance(denom, (int, float)) and denom == 0:
            raise ValueError("Cannot divide by zero")
        new_df[column] = s1 / denom
    elif op == "power":
        if s2 is None: s2 = 2.0
        new_df[column] = s1 ** s2
    elif op == "log":
        new_df[column] = np.log1p(np.maximum(s1.fillna(0), 0))
    elif op == "sqrt":
        new_df[column] = np.sqrt(np.maximum(s1.fillna(0), 0))
    elif op == "abs":
        new_df[column] = s1.abs()
    else:
        raise ValueError(f"Unsupported operation '{operation}'")

    return new_df
