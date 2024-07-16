import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars_pipeline.plot import KDE2dPlot, ScatterPlot


class TestRationalPlots(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 1000),
                "cat2": np.random.choice(["X", "Y"], 1000),
                "num1": np.random.randn(1000),
                "num2": np.random.randn(1000),
            },
            schema={
                "cat1": pl.Categorical,
                "cat2": pl.Categorical,
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

    def test_scatter_plot_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = ScatterPlot(hue="cat1", size="num1", style="cat2")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df)

    def test_kde2d_plot_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = KDE2dPlot(hue="cat1")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.df)
