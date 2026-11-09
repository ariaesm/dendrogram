"""Dendrogram rendering with matplotlib (headless Agg backend)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from .core import Result, cut_height  # noqa: E402


def save_dendrogram(
    result: Result,
    path: str | Path,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    truncate: int | None = None,
    dpi: int = 150,
) -> Path:
    """Draw the dendrogram coloured by the chosen k and save it as an image."""
    path = Path(path)
    n = len(result.linkage) + 1
    threshold = cut_height(result.linkage, result.k)
    fig_w = min(max(8.0, n * 0.18), 24.0) if truncate is None else 10.0
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    kwargs: dict = {"color_threshold": threshold, "above_threshold_color": "#888888", "ax": ax}
    if truncate is not None:
        kwargs.update(truncate_mode="lastp", p=truncate, show_contracted=True)
    elif labels is not None and n <= 150:
        kwargs.update(labels=list(labels), leaf_font_size=7)
    else:
        kwargs.update(no_labels=True)
    dendrogram(result.linkage, **kwargs)
    ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1)
    ax.text(
        ax.get_xlim()[1],
        threshold,
        f" k = {result.k}",
        color="#333333",
        va="bottom",
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Merge distance")
    ax.set_title(title or f"Hierarchical clustering ({n} rows, k = {result.k})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
