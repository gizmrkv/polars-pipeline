import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars_pipeline.plot import BoxPlot, ViolinPlot


class TestCategoricalPlots(unittest.TestCase):
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

        self.has_null_df = self.df.with_columns(
            [
                pl.Series(
                    col,
                    [val if random.random() > 0.1 else None for val in self.df[col]],
                    dtype=dtype,
                )
                for col, dtype in self.df.schema.items()
            ]
        )

    def test_boxplot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = BoxPlot(hue="cat1")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.has_null_df)

    def test_violinplot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot = ViolinPlot(hue="cat1")
            plot.log_dir = Path(tmpdir)
            plot.fit(self.has_null_df)
