import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars_pipeline.plot import CorrelationHeatmap, CountHeatmap
from sklearn.datasets import make_classification


class TestMatrixPlots(unittest.TestCase):
    def setUp(self):
        X_np, y_np = make_classification(n_samples=1000, n_classes=2, random_state=42)
        X = pl.from_numpy(X_np, schema=[f"feature_{i}" for i in range(X_np.shape[1])])
        y = pl.from_numpy(y_np, schema=["target"])
        self.df_corr = pl.concat([X, y], how="horizontal")
        self.df_corr = self.df_corr.with_columns(
            [
                pl.Series(
                    col,
                    [
                        val if random.random() > 0.1 else None
                        for val in self.df_corr[col]
                    ],
                    dtype=dtype,
                )
                for col, dtype in self.df_corr.schema.items()
            ]
        )

        self.df_count = pl.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 1000),
                "cat2": np.random.choice(["X", "Y"], 1000),
                "cat3": np.random.choice(["M", "N", "O"], 1000),
                "cat4": np.random.choice(["P", "Q", "R", "S"], 1000),
            },
            schema={
                "cat1": pl.Categorical,
                "cat2": pl.Categorical,
                "cat3": pl.Categorical,
                "cat4": pl.Categorical,
            },
        )
        self.df_count = self.df_count.with_columns(
            [
                pl.Series(
                    col,
                    [
                        val if random.random() > 0.1 else None
                        for val in self.df_count[col]
                    ],
                    dtype=dtype,
                )
                for col, dtype in self.df_count.schema.items()
            ]
        )

    def test_corr_heatmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = CorrelationHeatmap()
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df_corr)

    def test_count_heatmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = CountHeatmap()
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df_count)
