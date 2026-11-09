"""``dendro`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .core import LINKAGE_METHODS, METRICS, SCALERS, DendroError, attach_labels, run


def _csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dendro",
        description=(
            "Hierarchical clustering for any CSV: saves a dendrogram PNG, picks the number "
            "of clusters by silhouette score (or uses --k) and writes the CSV back with a "
            "cluster column."
        ),
    )
    p.add_argument("csv", type=Path, help="input CSV file")
    p.add_argument("-c", "--columns", type=_csv_list, help="comma-separated feature columns")
    p.add_argument(
        "-x", "--exclude", type=_csv_list, default=[], help="numeric columns to leave out"
    )
    p.add_argument("--id-column", help="column used as leaf labels in the dendrogram")
    p.add_argument("--sep", default=",", help="CSV delimiter (default: ',')")
    p.add_argument(
        "--scale", choices=SCALERS, default="standard", help="feature scaling (default: standard)"
    )
    p.add_argument("-m", "--method", choices=LINKAGE_METHODS, default="ward", help="linkage")
    p.add_argument("--metric", choices=METRICS, default="euclidean", help="distance metric")
    p.add_argument("-k", "--k", type=int, help="number of clusters (default: auto by silhouette)")
    p.add_argument("--k-min", type=int, default=2, help="smallest k to try (default: 2)")
    p.add_argument("--k-max", type=int, default=10, help="largest k to try (default: 10)")
    p.add_argument("-o", "--output", type=Path, help="output CSV (default: <input>_clustered.csv)")
    p.add_argument("--cluster-column", default="cluster", help="name of the added column")
    p.add_argument("-p", "--plot", type=Path, help="dendrogram image (default: <input>.png)")
    p.add_argument("--no-plot", action="store_true", help="skip the dendrogram image")
    p.add_argument(
        "--truncate", type=int, metavar="P", help="only show the last P merges (large datasets)"
    )
    p.add_argument("--title", help="plot title")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        df = pd.read_csv(args.csv, sep=args.sep)
        exclude = list(args.exclude)
        if args.id_column:
            if args.id_column not in df.columns:
                raise DendroError(f"unknown id column {args.id_column!r}")
            if args.id_column not in exclude:
                exclude.append(args.id_column)
        result = run(
            df,
            columns=args.columns,
            exclude=exclude,
            scaling=args.scale,
            method=args.method,
            metric=args.metric,
            k=args.k,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    except (DendroError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Rows: {result.rows_used} used, {result.rows_dropped} skipped (missing values)")
    print(f"Features ({len(result.columns)}): {', '.join(result.columns)}")
    print(f"Scaling: {args.scale} | linkage: {args.method} | metric: {args.metric}")
    if result.selection:
        print("Silhouette by k:")
        for k, s in result.selection.scores.items():
            marker = "  <- best" if k == result.k else ""
            print(f"  k={k:<3d} {s:.3f}{marker}")
    sil = f"{result.silhouette:.3f}" if result.silhouette is not None else "n/a"
    source = "auto" if result.selection else "user"
    print(f"Clusters: k={result.k} ({source}), silhouette={sil}")

    labelled = attach_labels(df, result, args.cluster_column)
    sizes = labelled[args.cluster_column].value_counts().sort_index()
    print("Cluster sizes: " + ", ".join(f"{c}: {n}" for c, n in sizes.items()))

    out = args.output or args.csv.with_name(f"{args.csv.stem}_clustered.csv")
    labelled.to_csv(out, index=False)
    print(f"Wrote {out}")

    if not args.no_plot:
        from .plot import save_dendrogram  # imported lazily: matplotlib is slow to load

        mask = df[result.columns].notna().all(axis=1)
        leaf_labels = df.loc[mask, args.id_column].astype(str).tolist() if args.id_column else None
        png = args.plot or args.csv.with_suffix(".png")
        save_dendrogram(result, png, leaf_labels, args.title, args.truncate)
        print(f"Wrote {png}")
    return 0
