import uuid
from typing import Iterable

import polars as pl

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class NullPredictor(Transformer):
    def __init__(
        self, model: Transformer, *, target: str, exclude: Iterable[str] | None = None
    ):
        self.model = model
        self.target = target
        self.exclude = exclude or []

    def fit(self, X: FrameType, y: FrameType | None = None):
        if log_dir := self.log_dir:
            self.model.log_dir = log_dir

        X_fill = X.filter(pl.col(self.target).is_not_null())
        y = X_fill.select(pl.col(self.target))
        X = X_fill.drop(self.target, *self.exclude)
        self.model.fit(X, y)

    def transform(self, X: FrameType) -> FrameType:
        if log_dir := self.log_dir:
            self.model.log_dir = log_dir

        # Add index to restore the order later
        index_name = str(uuid.uuid4())
        X = X.with_row_index(index_name)

        # Predict rows with missing target feature
        X_null = X.filter(pl.col(self.target).is_null()).drop(self.target)
        y_pred = self.model.transform(X_null.drop(index_name, *self.exclude))

        # Fill the missing values
        X_fill = X.filter(pl.col(self.target).is_not_null())
        X_pred = pl.concat([X_null, y_pred], how="horizontal")
        X_filled = (
            pl.concat([X_pred.select(X_fill.columns), X_fill])
            .sort(index_name)
            .drop(index_name)
        )
        return X_filled
