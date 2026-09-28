"""Small public datasets bundled with scikit-learn, as tidy DataFrames.

Both are built into scikit-learn (no download, no license question) and are
under 600 rows, so the whole EDA run finishes in seconds on a CPU.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_diabetes, load_wine


def load_wine_dataset() -> tuple[pd.DataFrame, str]:
    """Classification: 178 Italian wines, 13 chemical measurements, 3 cultivars."""
    data = load_wine(as_frame=True)
    df = data.frame.rename(columns={"target": "cultivar"})
    df["cultivar"] = df["cultivar"].map(dict(enumerate(data.target_names)))
    return df, "cultivar"


def load_diabetes_dataset() -> tuple[pd.DataFrame, str]:
    """Regression: 442 diabetes patients, 10 baseline measurements, disease progression target."""
    data = load_diabetes(as_frame=True)
    df = data.frame.rename(columns={"target": "disease_progression"})
    return df, "disease_progression"
