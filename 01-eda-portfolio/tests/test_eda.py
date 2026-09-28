import numpy as np
import pandas as pd
import pytest

from eda_portfolio import datasets, eda


@pytest.fixture
def toy_df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, None],
            "b": [10, 20, 30, 40, 50],
            "label": ["x", "x", "y", "y", "y"],
        }
    )


def test_summary_stats_counts_and_missing(toy_df):
    stats = eda.summary_stats(toy_df)
    assert stats.loc["a", "count"] == 4
    assert stats.loc["a", "missing"] == 1
    assert stats.loc["a", "missing_pct"] == 20.0
    assert stats.loc["b", "missing"] == 0
    assert stats.loc["b", "mean"] == 30.0
    # non-numeric column has no numeric stats
    assert pd.isna(stats.loc["label", "mean"])


def test_missing_report_only_lists_columns_with_gaps(toy_df):
    report = eda.missing_report(toy_df)
    assert list(report.index) == ["a"]
    assert report.loc["a", "missing"] == 1


def test_missing_report_empty_when_no_gaps():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert eda.missing_report(df).empty


def test_outlier_report_detects_planted_outlier():
    # 19 values tightly clustered around 10, one planted outlier at 1000.
    values = [10 + i * 0.1 for i in range(19)] + [1000.0]
    df = pd.DataFrame({"v": values})
    report = eda.outlier_report(df)
    assert report.loc["v", "outliers"] == 1


def test_outlier_report_no_outliers_on_uniform_data():
    df = pd.DataFrame({"v": list(range(20))})
    report = eda.outlier_report(df)
    assert report.loc["v", "outliers"] == 0


def test_correlation_report_finds_perfect_linear_pair():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y_perfect": x * 2 + 1, "noise": rng.normal(size=200)})
    top = eda.correlation_report(df, top_n=1)
    assert {top.loc[0, "feature_a"], top.loc[0, "feature_b"]} == {"x", "y_perfect"}
    assert top.loc[0, "correlation"] == pytest.approx(1.0, abs=1e-9)


def test_correlation_report_no_duplicate_pairs():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "c": [1, 1, 2, 2]})
    top = eda.correlation_report(df, top_n=10)
    pairs = list(zip(top["feature_a"], top["feature_b"]))
    assert len(pairs) == len(set(pairs))
    assert len(pairs) == 3  # 3 choose 2, each pair once


def test_class_balance_sums_to_100():
    target = pd.Series(["a", "a", "a", "b"])
    balance = eda.class_balance(target)
    assert balance.loc["a", "count"] == 3
    assert balance["pct"].sum() == pytest.approx(100.0)


def test_wine_dataset_shape_and_target():
    df, target = datasets.load_wine_dataset()
    assert df.shape == (178, 14)
    assert target == "cultivar"
    assert set(df[target].unique()) == {"class_0", "class_1", "class_2"}


def test_diabetes_dataset_shape_and_target():
    df, target = datasets.load_diabetes_dataset()
    assert df.shape == (442, 11)
    assert target == "disease_progression"
    assert df[target].isna().sum() == 0
