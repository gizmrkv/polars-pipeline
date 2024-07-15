import math
from typing import Dict, Sequence

import polars as pl
from polars import LazyFrame
from polars._typing import FrameType

from polars_pipeline.transformer import Transformer


class StandardScaler(Transformer):
    def __init__(self, columns: str | Sequence[str]):
        self.columns = [columns] if isinstance(columns, str) else columns

        self.mean_values: Dict[str, float] = {}
        self.std_values: Dict[str, float] = {}

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame is not supported")

        self.mean_values.clear()
        self.std_values.clear()

        for col in self.columns:
            self.mean_values[col] = float(X.select(col).mean().row(0)[0])
            self.std_values[col] = float(X.select(col).std().row(0)[0])

            if math.isclose(self.std_values[col], 0.0):
                raise ZeroDivisionError(f"Columns have zero diff: {col}")

    def transform(self, X: FrameType) -> FrameType:
        for col in self.columns:
            X = X.with_columns(
                ((pl.col(col) - self.mean_values[col]) / self.std_values[col])
            )

        return X
