from typing import Any, Callable, Dict, List

import lightgbm as lgb
import numpy as np
import polars as pl
from polars import LazyFrame

from polars_pipeline.exception import (
    ColumnsMismatchError,
    LazyFrameNotSupportedError,
    NotFittedError,
)
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class LightGBM(Transformer):
    def __init__(
        self,
        params: Dict[str, Any],
        *,
        train_fn: Callable[[lgb.Dataset], lgb.Booster] | None = None,
        predict_fn: Callable[[lgb.Booster, np.ndarray], np.ndarray] | None = None,
    ):
        def default_train_fn(data: lgb.Dataset) -> lgb.Booster:
            return lgb.train(self.params, data)  # type: ignore

        def default_predict_fn(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
            return booster.predict(X)  # type: ignore

        self.params = params
        self.train_fn = train_fn or default_train_fn
        self.predict_fn = predict_fn or default_predict_fn
        self.booster: lgb.Booster | None = None
        self.X_columns: List[str] | None = None
        self.y_column: str | None = None

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame) or isinstance(y, LazyFrame):
            raise LazyFrameNotSupportedError(self.__class__.__name__, self.fit.__name__)

        if y is None:
            raise ValueError("y cannot be None")

        if len(y.columns) > 1:
            raise ValueError("y should have only one column")

        self.X_columns = X.columns
        self.y_column = y.columns[0]

        X_np = X.to_numpy().squeeze()
        y_np = y.to_numpy().squeeze()

        data = lgb.Dataset(X_np, label=y_np)
        self.booster = self.train_fn(data)

    def transform(self, X: FrameType) -> FrameType:
        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.transform.__name__
            )

        if self.booster is None or self.X_columns is None or self.y_column is None:
            raise NotFittedError(self.__class__.__name__)

        if set(self.X_columns) != set(X.columns):
            raise ColumnsMismatchError(
                self.__class__.__name__, X.columns, self.X_columns
            )

        X_np = X.to_numpy().squeeze()
        pred = self.predict_fn(self.booster, X_np)
        assert isinstance(pred, np.ndarray)

        if self.params["objective"] == "multiclass":
            return pl.from_numpy(
                pred, schema=[f"{self.y_column}_{i}" for i in range(pred.shape[1])]
            )
        else:
            return pl.from_numpy(pred, schema=[self.y_column])
