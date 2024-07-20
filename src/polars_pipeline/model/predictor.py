from typing import Iterable

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class Predictor(Transformer):
    def __init__(self, predicator: Transformer, *, target: str | Iterable[str]):
        self.predicator = predicator
        self.target = target

    def fit(self, X: FrameType, y: FrameType | None = None):
        if log_dir := self.log_dir:
            self.predicator.log_dir = log_dir

        y = X.select(self.target)
        X = X.drop(self.target)
        self.predicator.fit(X, y)

    def transform(self, X: FrameType) -> FrameType:
        if log_dir := self.log_dir:
            self.predicator.log_dir = log_dir

        return self.predicator.transform(X.drop(self.target))
