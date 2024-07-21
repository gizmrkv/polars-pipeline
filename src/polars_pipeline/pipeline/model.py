from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal

if TYPE_CHECKING:
    from polars_pipeline import Pipeline


import lightgbm as lgb
import numpy as np
from polars import DataFrame
from polars._typing import IntoExpr
from sklearn.model_selection import BaseCrossValidator

from polars_pipeline.model import LightGBM, NullPredictor, Predictor, Stacker
from polars_pipeline.transformer import Transformer


class ModelNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def predict(self, model: Transformer, *, target: str | Iterable[str]) -> "Pipeline":
        return self.pipeline.pipe(Predictor(model, target=target))

    def predict_null(
        self, model: Transformer, *, target: str, exclude: Iterable[str] | None = None
    ) -> "Pipeline":
        return self.pipeline.pipe(NullPredictor(model, target=target, exclude=exclude))

    def lightgbm(
        self,
        params: Dict[str, Any],
        *,
        train_fn: Callable[[lgb.Dataset], lgb.Booster] | None = None,
        predict_fn: Callable[[lgb.Booster, np.ndarray], np.ndarray] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBM(params, train_fn=train_fn, predict_fn=predict_fn)
        )

    def stack(
        self,
        model: Transformer,
        *,
        fold: BaseCrossValidator,
        aggs: Iterable[IntoExpr] | Literal["mean"] = "mean",
        groups: str | None = None,
        metrics_fn: Callable[[DataFrame, DataFrame], Dict[str, Any]] | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Stacker(model, fold=fold, aggs=aggs, groups=groups, metrics_fn=metrics_fn)
        )
