import math
from typing import Dict, Sequence, Tuple

import polars as pl
from polars import LazyFrame
from polars._typing import FrameType

from polars_pipeline.transformer import Transformer


class RobustScaler(Transformer):
    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
    ):
        q1, q3 = quantile_range
        if q1 >= q3:
            raise ValueError(
                f"quantile_range must be in increasing order: {quantile_range}"
            )

        self.columns = [columns] if isinstance(columns, str) else columns
        self.q1 = q1
        self.q3 = q3

        self.median_values: Dict[str, float] = {}
        self.iqr_values: Dict[str, float] = {}

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame is not supported")

        self.median_values.clear()
        self.iqr_values.clear()

        for col in self.columns:
            self.median_values[col] = float(X.select(col).median().row(0)[0])
            self.iqr_values[col] = float(
                X.select(
                    pl.col(col).quantile(self.q3) - pl.col(col).quantile(self.q1)
                ).row(0)[0]
            )

            if math.isclose(self.iqr_values[col], 0.0):
                raise ZeroDivisionError(f"Columns have zero iqr: {col}")

    def transform(self, X: FrameType) -> FrameType:
        for col in self.columns:
            X = X.with_columns(
                ((pl.col(col) - self.median_values[col]) / self.iqr_values[col])
            )

        return X
