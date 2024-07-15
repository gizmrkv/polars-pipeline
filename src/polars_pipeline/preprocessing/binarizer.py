from typing import Sequence

import polars as pl
from polars._typing import FrameType

from polars_pipeline.transformer import Transformer


class Binarizer(Transformer):
    def __init__(
        self, columns: str | Sequence[str] | None = None, *, threshold: float = 0.5
    ):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.threshold = threshold

    def run(self, frame: FrameType) -> FrameType:
        columns = self.columns or frame.columns
        return frame.with_columns(
            [pl.col(col).gt(self.threshold).cast(pl.Int32) for col in columns]
        )
