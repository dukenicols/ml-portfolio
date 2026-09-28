"""Dataset-agnostic exploratory data analysis functions.

Every function takes a pandas DataFrame (and sometimes a Series) and returns
a plain DataFrame or dict, so the same code runs against any tabular dataset
loaded by eda_portfolio.datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count, missing count/pct, mean, std, min, max.

    Non-numeric columns get NaN for the numeric-only fields.
    """
    numeric = df.select_dtypes("number")
    stats = pd.DataFrame(
        {
            "count": df.count(),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        }
    )
    for col in ("mean", "std", "min", "max"):
        stats[col] = getattr(numeric, col)().reindex(df.columns)
    return stats


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columns with at least one missing value, sorted worst first."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {"missing": missing, "missing_pct": (missing / len(df) * 100).round(2)}
    )


def outlier_report(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Count outliers per numeric column using the IQR rule.

    A value is an outlier if it falls outside [Q1 - k*IQR, Q3 + k*IQR].
    k=1.5 is Tukey's standard fence.
    """
    numeric = df.select_dtypes("number")
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        count = int(((series < low) | (series > high)).sum())
        rows.append(
            {
                "column": col,
                "outliers": count,
                "outlier_pct": round(count / len(series) * 100, 2) if len(series) else 0.0,
                "lower_fence": round(low, 3),
                "upper_fence": round(high, 3),
            }
        )
    return pd.DataFrame(rows).set_index("column").sort_values("outliers", ascending=False)


def correlation_report(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Strongest pairwise absolute Pearson correlations among numeric columns."""
    numeric = df.select_dtypes("number")
    corr = numeric.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool))
    # stack() no longer drops NaN by default (pandas >= 2.1 future_stack behavior).
    pairs = corr.where(~mask).stack().dropna().reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    # Matrix position gives feature_a > feature_b; swap so pairs read alphabetically.
    swap = pairs["feature_a"] > pairs["feature_b"]
    pairs.loc[swap, ["feature_a", "feature_b"]] = pairs.loc[swap, ["feature_b", "feature_a"]].values
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return (
        pairs.sort_values("abs_correlation", ascending=False)
        .drop(columns="abs_correlation")
        .head(top_n)
        .reset_index(drop=True)
    )


def class_balance(target: pd.Series) -> pd.DataFrame:
    """Value counts and percentage share for a categorical/target column."""
    counts = target.value_counts()
    return pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)})
