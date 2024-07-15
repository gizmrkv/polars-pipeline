import math
from typing import Dict, Sequence

import polars as pl
from polars import LazyFrame

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class MinMaxScaler(Transformer):
    def __init__(self, columns: str | Sequence[str]):
        self.columns = [columns] if isinstance(columns, str) else columns

        self.max_values: Dict[str, float] = {}
        self.min_values: Dict[str, float] = {}
        self.diff_values: Dict[str, float] = {}

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame is not supported")

        self.max_values.clear()
        self.min_values.clear()
        self.diff_values.clear()

        for col in self.columns:
            self.max_values[col] = float(X.select(col).max().row(0)[0])
            self.min_values[col] = float(X.select(col).min().row(0)[0])
            self.diff_values[col] = self.max_values[col] - self.min_values[col]

            if math.isclose(self.diff_values[col], 0.0):
                raise ZeroDivisionError(f"Columns have zero diff: {col}")

    def transform(self, X: FrameType) -> FrameType:
        for col in self.columns:
            X = X.with_columns(
                ((pl.col(col) - self.min_values[col]) / self.diff_values[col])
            )

        return X
