"""Regenerate examples/synthetic_customers.csv (synthetic data, fixed random seed).

Three well-separated groups are drawn with scikit-learn's ``make_blobs`` and then mapped to
plausible-looking ranges. The ``segment`` column holds the true group so you can check how
well the clustering recovers it; it is a string, so dendro ignores it automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=60, centers=3, n_features=3, cluster_std=1.2, random_state=7)
df = pd.DataFrame(
    {
        "customer_id": [f"C{i:03d}" for i in range(1, len(x) + 1)],
        "segment": np.array(["A", "B", "C"])[y],
        "annual_spend": np.round(1500 + 120 * x[:, 0], 0),
        "visits_per_month": np.round(np.clip(8 + 0.6 * x[:, 1], 0.5, None), 1),
        "avg_basket": np.round(40 + 3.5 * x[:, 2], 2),
    }
)
out = Path(__file__).with_name("synthetic_customers.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows)")
