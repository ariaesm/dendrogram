import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from dendro.core import (
    DendroError,
    attach_labels,
    choose_k,
    cluster_linkage,
    cut,
    cut_height,
    run,
    scale,
    select_columns,
    validate_options,
)


@pytest.fixture
def blobs_df():
    x, y = make_blobs(n_samples=45, centers=3, n_features=2, cluster_std=0.5, random_state=0)
    return pd.DataFrame({"name": [f"r{i}" for i in range(45)], "a": x[:, 0], "b": x[:, 1], "y": y})


def test_select_columns_auto_skips_text_and_bool():
    df = pd.DataFrame({"s": ["x"], "i": [1], "f": [1.5], "flag": [True]})
    assert select_columns(df) == ["i", "f"]
    assert select_columns(df, exclude=["i"]) == ["f"]
    assert select_columns(df, columns=["f"]) == ["f"]


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"columns": ["nope"]}, "unknown"),
        ({"columns": ["s"]}, "not numeric"),
        ({"exclude": ["i"]}, "no numeric"),
    ],
)
def test_select_columns_errors(kwargs, msg):
    df = pd.DataFrame({"s": ["x"], "i": [1]})
    with pytest.raises(DendroError, match=msg):
        select_columns(df, **kwargs)


def test_scale():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    std = scale(x, "standard")
    assert np.allclose(std[:, 0], [-1, 1]) and np.allclose(std[:, 1], 0)  # constant col -> 0
    assert np.allclose(scale(x, "minmax")[:, 0], [0, 1])
    assert np.array_equal(scale(x, "none"), x)
    with pytest.raises(DendroError):
        scale(x, "log")


def test_validate_options():
    validate_options("average", "cosine")
    with pytest.raises(DendroError, match="requires metric"):
        validate_options("ward", "cityblock")
    with pytest.raises(DendroError):
        validate_options("magic", "euclidean")


def test_choose_k_recovers_three_blobs(blobs_df):
    x = scale(blobs_df[["a", "b"]].to_numpy())
    z = cluster_linkage(x)
    sel = choose_k(x, z)
    assert sel.best_k == 3
    assert set(sel.scores) == set(range(2, 11))
    assert all(-1 <= s <= 1 for s in sel.scores.values())


def test_run_auto_matches_truth(blobs_df):
    res = run(blobs_df, exclude=["y"])
    assert res.k == 3 and res.columns == ["a", "b"]
    assert adjusted_rand_score(blobs_df["y"], res.labels) > 0.9
    assert res.silhouette > 0.7


def test_run_user_k_and_other_linkages(blobs_df):
    for method, metric in [
        ("average", "cityblock"),
        ("complete", "cosine"),
        ("single", "chebyshev"),
    ]:
        res = run(blobs_df, exclude=["y"], method=method, metric=metric, k=4)
        assert res.selection is None
        assert len(np.unique(res.labels)) == 4


def test_run_bad_k(blobs_df):
    with pytest.raises(DendroError):
        run(blobs_df, exclude=["y"], k=1)
    with pytest.raises(DendroError):
        run(blobs_df.head(2), exclude=["y"])


def test_cut_height_separates_k(blobs_df):
    x = scale(blobs_df[["a", "b"]].to_numpy())
    z = cluster_linkage(x)
    for k in (2, 3, 5):
        assert len(np.unique(fcluster(z, cut_height(z, k), criterion="distance"))) == k
        assert len(np.unique(cut(z, k))) == k


def test_missing_values_are_skipped_and_marked_na(blobs_df):
    df = blobs_df.copy()
    df.loc[[0, 5], "a"] = np.nan
    res = run(df, exclude=["y"])
    assert (res.rows_used, res.rows_dropped) == (43, 2)
    out = attach_labels(df, res, "grp")
    assert out["grp"].isna().sum() == 2
    assert out["grp"].dtype == "Int64"
    assert list(out.columns) == [*df.columns, "grp"]
