from typing import Iterable

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class Predictor(Transformer):
    def __init__(self, model: Transformer, *, target: str | Iterable[str]):
        self.model = model
        self.target = target

    def fit(self, X: FrameType, y: FrameType | None = None):
        if log_dir := self.log_dir:
            self.model.log_dir = log_dir

        y = X.select(self.target)
        X = X.drop(self.target)
        self.model.fit(X, y)

    def transform(self, X: FrameType) -> FrameType:
        if log_dir := self.log_dir:
            self.model.log_dir = log_dir

        return self.model.transform(X.drop(self.target))
