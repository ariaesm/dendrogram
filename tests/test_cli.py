from pathlib import Path

import pandas as pd

from dendro.cli import main

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "synthetic_customers.csv"


def test_cli_end_to_end(tmp_path, capsys):
    out_csv, png = tmp_path / "out.csv", tmp_path / "plot.png"
    code = main([str(SAMPLE), "--id-column", "customer_id", "-o", str(out_csv), "-p", str(png)])
    assert code == 0
    text = capsys.readouterr().out
    assert "Clusters: k=3 (auto)" in text
    df = pd.read_csv(out_csv)
    assert df["cluster"].nunique() == 3
    # the synthetic sample has 3 true segments; each must map to exactly one cluster
    assert (pd.crosstab(df["segment"], df["cluster"]) > 0).sum(axis=1).eq(1).all()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_user_k_truncate_and_defaults(tmp_path, capsys):
    src = tmp_path / "data.csv"
    src.write_bytes(SAMPLE.read_bytes())
    code = main([str(src), "-k", "4", "-m", "average", "--metric", "cityblock", "--truncate", "10"])
    assert code == 0
    assert "k=4 (user)" in capsys.readouterr().out
    assert (tmp_path / "data_clustered.csv").exists()
    assert (tmp_path / "data.png").exists()


def test_cli_no_plot_and_custom_column(tmp_path):
    out = tmp_path / "o.csv"
    assert (
        main([str(SAMPLE), "--no-plot", "-o", str(out), "--cluster-column", "grp", "-k", "2"]) == 0
    )
    assert "grp" in pd.read_csv(out).columns
    assert not SAMPLE.with_suffix(".png").exists()


def test_cli_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert main([str(SAMPLE), "-c", "segment"]) == 2
    assert main([str(SAMPLE), "-m", "ward", "--metric", "cosine"]) == 2
    assert main([str(SAMPLE), "--id-column", "nope"]) == 2
    err = capsys.readouterr().err
    assert "not numeric" in err and "requires metric" in err
