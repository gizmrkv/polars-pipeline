import unittest

import polars as pl
from polars.testing import assert_frame_equal
from polars_pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1, None, 3, 4, None],
                "c": [0.1, 0.2, 0.3, 0.4, 0.5],
                "d": [None, 0.2, None, 0.4, 0.5],
                "e": [True, False, True, False, True],
                "f": [True, None, None, None, True],
                "g": ["foo", "bar", "ham", "spam", "jam"],
                "h": ["foo", "bar", None, "spam", "jam"],
            }
        )

    def test_select(self):
        pipeline = Pipeline().select("a", "b")
        out = pipeline.transform(self.df)
        assert_frame_equal(out, self.df.select(["a", "b"]))

    def test_with_columns(self):
        pipeline = Pipeline().with_columns(pl.col("a").alias("a2"))
        out = pipeline.transform(self.df)
        assert_frame_equal(
            out,
            self.df.with_columns(pl.col("a").alias("a2")),
        )

    def test_drop(self):
        pipeline = Pipeline().drop("a")
        out = pipeline.transform(self.df)
        assert_frame_equal(out, self.df.drop("a"))
