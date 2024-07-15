from typing import Dict, Sequence

import polars as pl
import polars.selectors as sc
from polars import DataFrame, LazyFrame

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class LabelEncoder(Transformer):
    def __init__(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        maintain_order: bool = False,
    ):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.maintain_order = maintain_order

        self.mappings: Dict[str, DataFrame] = {}

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame is not supported")

        self.mappings.clear()
        x = X.select(self.columns or sc.categorical())
        for col in x.columns:
            mapping = x.select(col).unique(maintain_order=self.maintain_order)
            mapping = mapping.with_columns(
                pl.arange(0, len(mapping), dtype=pl.Int64).alias("label")
            )
            self.mappings[col] = mapping

    def transform(self, X: FrameType) -> FrameType:
        for col, mapping in self.mappings.items():
            if isinstance(X, DataFrame):
                X = X.join(mapping, on=col, how="left", coalesce=True)
            else:
                X = X.join(mapping.lazy(), on=col, how="left", coalesce=True)
            X = X.drop(col).rename({"label": col})

        return X
