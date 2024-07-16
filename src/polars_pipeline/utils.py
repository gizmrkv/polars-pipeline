from typing import List

import polars as pl

from .typing import FrameType


def numerical_columns(frame: FrameType) -> List[str]:
    return frame.head(1).select(pl.col(pl.Float32, pl.Float64, pl.Decimal)).columns


def categorical_columns(frame: FrameType) -> List[str]:
    return frame.head(1).select(pl.col(pl.Categorical, pl.Enum, pl.Boolean)).columns
