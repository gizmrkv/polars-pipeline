import json
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Literal

import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame
from polars._typing import IntoExpr
from sklearn.model_selection import BaseCrossValidator

from polars_pipeline.exception import (
    LazyFrameNotSupportedError,
    NotFittedError,
    TargetRequiredError,
)
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType
from polars_pipeline.utils import list_of_dict_to_dict_of_list


class Stacker(Transformer):
    def __init__(
        self,
        model: Transformer,
        *,
        fold: BaseCrossValidator,
        aggs: Iterable[IntoExpr] | Literal["mean"] = "mean",
        groups: str | None = None,
        metrics_fn: Callable[[DataFrame, DataFrame], Dict[str, Any]] | None = None,
    ):
        if aggs == "mean":
            aggs = [pl.all().mean()]

        self.model = model
        self.fold = fold
        self.groups = groups
        self.aggs = aggs
        self.metrics_fn = metrics_fn
        self.models: List[Transformer] = []
        self.valid_indexes: List[np.ndarray] = []

    def fit(self, X: FrameType, y: FrameType | None = None):
        if isinstance(X, LazyFrame) or isinstance(y, LazyFrame):
            raise LazyFrameNotSupportedError(self.__class__.__name__, self.fit.__name__)

        if y is None:
            raise TargetRequiredError(self.__class__.__name__)

        self.models.clear()
        self.valid_indexes.clear()
        metrics_list = []
        for i, (train_idx, valid_idx) in enumerate(
            self.fold.split(X, y, groups=self.groups)
        ):
            X_train = X.select(pl.all().gather(train_idx))
            y_train = y.select(pl.all().gather(train_idx))
            X_valid = X.select(pl.all().gather(valid_idx))
            y_valid = y.select(pl.all().gather(valid_idx))
            model = deepcopy(self.model)
            if log_dir := self.log_dir:
                model.log_dir = log_dir / f"fold_{i}"

            model.fit(X_train, y_train)
            self.models.append(model)
            self.valid_indexes.append(valid_idx)

            if self.metrics_fn and self.log_dir:
                metrics = self.metrics_fn(y_valid, model.transform(X_valid))
                metrics_list.append(metrics)

        if len(metrics_list) > 0 and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "metrics.json", "w") as f:
                json.dump(list_of_dict_to_dict_of_list(metrics_list), f, indent=4)

    def transform(self, X: FrameType) -> FrameType:
        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.transform.__name__
            )

        if not self.models:
            raise NotFittedError(self.__class__.__name__)

        if log_dir := self.log_dir:
            for i, model in enumerate(self.models):
                model.log_dir = log_dir / f"fold_{i}"

        index_name = str(uuid.uuid4())
        preds = [model.transform(X) for model in self.models]
        pred_catted: DataFrame = pl.concat(
            [pred.with_row_index(index_name) for pred in preds],
            how="vertical",
        )
        pred = (
            pred_catted.group_by(index_name)
            .agg(*self.aggs)
            .sort(index_name)
            .drop(index_name)
        )
        return pred  # type: ignore

    def fit_transform(self, X: FrameType, y: FrameType | None = None) -> FrameType:
        self.fit(X, y)
        valid_index = np.concatenate(self.valid_indexes)
        preds = [
            model.transform(X.select(pl.all().gather(valid_idx)))
            for model, valid_idx in zip(self.models, self.valid_indexes)
        ]
        index_name = str(uuid.uuid4())
        pred = (
            pl.concat(
                [
                    pl.concat(preds, how="vertical"),  # type: ignore
                    pl.from_numpy(valid_index, schema=[index_name]),
                ],
                how="horizontal",
            )
            .sort(index_name)
            .drop(index_name)
        )
        return pred
