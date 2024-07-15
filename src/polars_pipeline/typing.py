from typing import TypeVar

from polars import DataFrame, LazyFrame

FrameType = TypeVar("FrameType", DataFrame, LazyFrame)
