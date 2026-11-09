"""Data preparation, hierarchical clustering and automatic choice of k.

Everything here is free of plotting and I/O so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

LINKAGE_METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
EUCLIDEAN_ONLY = {"centroid", "median", "ward"}
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "cosine", "correlation")
SCALERS = ("standard", "minmax", "none")


class DendroError(ValueError):
    """User-facing error (bad column names, too few rows, invalid options...)."""


def select_columns(
    df: pd.DataFrame, columns: list[str] | None = None, exclude: list[str] | None = None
) -> list[str]:
    """Return the feature columns: the given ones, or every numeric column minus ``exclude``."""
    exclude = exclude or []
    missing = [c for c in [*(columns or []), *exclude] if c not in df.columns]
    if missing:
        raise DendroError(f"unknown column(s): {', '.join(missing)}")
    if columns:
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise DendroError(f"column(s) are not numeric: {', '.join(non_numeric)}")
        chosen = [c for c in columns if c not in exclude]
    else:
        chosen = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
            and c not in exclude
        ]
    if not chosen:
        raise DendroError("no numeric columns to cluster on")
    return chosen


def scale(x: np.ndarray, how: str = "standard") -> np.ndarray:
    """Scale features column-wise. Constant columns become 0 instead of NaN."""
    if how not in SCALERS:
        raise DendroError(f"unknown scaling {how!r}; choose from {', '.join(SCALERS)}")
    x = np.asarray(x, dtype=float)
    if how == "none":
        return x
    if how == "standard":
        center, spread = x.mean(axis=0), x.std(axis=0)
    else:
        center, spread = x.min(axis=0), x.max(axis=0) - x.min(axis=0)
    spread = np.where(spread == 0, 1.0, spread)
    return (x - center) / spread


def validate_options(method: str, metric: str) -> None:
    if method not in LINKAGE_METHODS:
        raise DendroError(f"unknown linkage {method!r}; choose from {', '.join(LINKAGE_METHODS)}")
    if metric not in METRICS:
        raise DendroError(f"unknown metric {metric!r}; choose from {', '.join(METRICS)}")
    if method in EUCLIDEAN_ONLY and metric != "euclidean":
        raise DendroError(f"linkage {method!r} requires metric 'euclidean' (got {metric!r})")


def cluster_linkage(x: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    """SciPy linkage matrix for the rows of ``x``."""
    validate_options(method, metric)
    if len(x) < 3:
        raise DendroError("need at least 3 rows to cluster")
    return linkage(x, method=method, metric=metric)


def cut(z: np.ndarray, k: int) -> np.ndarray:
    """Flat cluster labels 1..k from a linkage matrix."""
    return fcluster(z, t=k, criterion="maxclust")


def cut_height(z: np.ndarray, k: int) -> float:
    """A distance threshold that separates exactly ``k`` clusters (midpoint between merges)."""
    n = len(z) + 1
    if not 2 <= k <= n:
        raise DendroError(f"k must be between 2 and {n}")
    heights = np.sort(z[:, 2])
    upper = heights[-(k - 1)]
    lower = heights[-k] if k < n else 0.0
    return float((upper + lower) / 2)


@dataclass
class KSelection:
    best_k: int
    scores: dict[int, float] = field(default_factory=dict)


def choose_k(
    x: np.ndarray, z: np.ndarray, metric: str = "euclidean", k_min: int = 2, k_max: int = 10
) -> KSelection:
    """Pick the k in ``[k_min, k_max]`` with the highest mean silhouette score.

    Silhouette is computed on the same distance metric used for clustering
    (``sqeuclidean`` is evaluated on plain euclidean distances).
    """
    n = len(x)
    k_max = min(k_max, n - 1)
    if k_min < 2 or k_min > k_max:
        raise DendroError(f"invalid k range [{k_min}, {k_max}] for {n} rows")
    sil_metric = "euclidean" if metric == "sqeuclidean" else metric
    dist = squareform(pdist(x, metric=sil_metric))
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = cut(z, k)
        n_labels = len(np.unique(labels))
        if n_labels != k or n_labels >= n:  # ties in merge heights can merge clusters
            continue
        scores[k] = float(silhouette_score(dist, labels, metric="precomputed"))
    if not scores:
        raise DendroError("could not evaluate any k; try a different k range or linkage")
    best = max(scores, key=lambda k: (scores[k], -k))  # ties -> fewer clusters
    return KSelection(best_k=best, scores=scores)


@dataclass
class Result:
    columns: list[str]
    rows_used: int
    rows_dropped: int
    linkage: np.ndarray
    labels: np.ndarray
    k: int
    selection: KSelection | None
    silhouette: float | None


def run(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    scaling: str = "standard",
    method: str = "ward",
    metric: str = "euclidean",
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 10,
) -> Result:
    """Full pipeline on a DataFrame. Rows with missing values in the features are skipped."""
    validate_options(method, metric)
    cols = select_columns(df, columns, exclude)
    mask = df[cols].notna().all(axis=1).to_numpy()
    x = scale(df.loc[mask, cols].to_numpy(dtype=float), scaling)
    z = cluster_linkage(x, method, metric)
    selection = None
    if k is None:
        selection = choose_k(x, z, metric, k_min, k_max)
        k = selection.best_k
    elif not 2 <= k < len(x):
        raise DendroError(f"k must be between 2 and {len(x) - 1}")
    labels = cut(z, k)
    sil_metric = "euclidean" if metric == "sqeuclidean" else metric
    sil = (
        float(silhouette_score(x, labels, metric=sil_metric))
        if 1 < len(np.unique(labels)) < len(x)
        else None
    )
    return Result(
        columns=cols,
        rows_used=int(mask.sum()),
        rows_dropped=int((~mask).sum()),
        linkage=z,
        labels=labels,
        k=k,
        selection=selection,
        silhouette=sil,
    )


def attach_labels(df: pd.DataFrame, result: Result, column: str = "cluster") -> pd.DataFrame:
    """Copy of ``df`` with a nullable integer cluster column (<NA> for skipped rows)."""
    out = df.copy()
    labels = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mask = df[result.columns].notna().all(axis=1)
    labels[mask] = result.labels
    out[column] = labels
    return out
