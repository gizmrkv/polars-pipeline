import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars_pipeline.plot import HistPlot, KDEPlot


class TestDistributionPlots(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame(
            {
                "cat": np.random.choice(["A", "B", "C"], 1000),
                "num1": np.random.randn(1000),
                "num2": np.random.randn(1000),
            },
            schema={
                "cat": pl.Categorical,
                "num1": pl.Float32,
                "num2": pl.Float32,
            },
        )
        self.df = self.df.with_columns(
            [
                pl.Series(
                    col,
                    [val if random.random() > 0.1 else None for val in self.df[col]],
                    dtype=dtype,
                )
                for col, dtype in self.df.schema.items()
            ]
        )

    def test_hist_plot_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = HistPlot(hue="cat")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df)

    def test_kde_plot_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = KDEPlot(hue="cat")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df)
