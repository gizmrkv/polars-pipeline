import uuid
from typing import Callable, Generic, Iterable, Sequence

import polars as pl
from polars import Expr

from polars_pipeline.typing import FrameType


@pl.api.register_dataframe_namespace("horizontal")
@pl.api.register_lazyframe_namespace("horizontal")
class Horizontal(Generic[FrameType]):
    def __init__(self, frame: FrameType):
        self.frame: FrameType = frame

    def agg(
        self,
        columns: Iterable[str],
        *,
        agg_fn: Callable[[Expr, Expr], Expr],
        name: str = "agg",
    ) -> FrameType:
        it = iter(columns)
        agg_column = pl.col(next(it))
        for e in it:
            agg_column = agg_fn(agg_column, pl.col(e))

        return self.frame.with_columns(agg_column.alias(name))

    def mean(self, columns: Sequence[str], *, name: str = "mean") -> FrameType:
        return self.sum(columns, name=name).with_columns(
            pl.col(name).truediv(len(columns))
        )

    def sum(self, columns: Iterable[str], *, name: str = "sum") -> FrameType:
        return self.agg(columns, agg_fn=lambda a, b: a + b, name=name)

    def prod(self, columns: Iterable[str], *, name: str = "prod") -> FrameType:
        return self.agg(columns, agg_fn=lambda a, b: a * b, name=name)

    def all(self, columns: Iterable[str], *, name: str = "all") -> FrameType:
        return self.agg(columns, agg_fn=lambda a, b: a & b, name=name)

    def any(self, columns: Iterable[str], *, name: str = "any") -> FrameType:
        return self.agg(columns, agg_fn=lambda a, b: a | b, name=name)

    def max(self, columns: Iterable[str], *, name: str = "max") -> FrameType:
        return self.agg(
            columns, agg_fn=lambda a, b: pl.when(a > b).then(a).otherwise(b), name=name
        )

    def min(self, columns: Iterable[str], *, name: str = "min") -> FrameType:
        return self.agg(
            columns, agg_fn=lambda a, b: pl.when(a < b).then(a).otherwise(b), name=name
        )

    def argmax(self, columns: Iterable[str], *, name: str = "argmax") -> FrameType:
        it = iter(columns)
        col = next(it)
        max_name = str(uuid.uuid4())
        frame = self.frame.with_columns(
            pl.lit(0).alias(name), pl.col(col).alias(max_name)
        )
        for i, col in enumerate(it, 1):
            frame = frame.with_columns(
                pl.when(pl.col(col) > pl.col(max_name))
                .then(i)
                .otherwise(pl.col(name))
                .alias(name),
                pl.when(pl.col(col) > pl.col(max_name))
                .then(pl.col(col))
                .otherwise(pl.col(max_name))
                .alias(max_name),
            )

        return frame.drop(max_name)

    def argmin(self, columns: Iterable[str], *, name: str = "argmin") -> FrameType:
        it = iter(columns)
        col = next(it)
        min_name = str(uuid.uuid4())
        frame = self.frame.with_columns(
            pl.lit(0).alias(name), pl.col(col).alias(min_name)
        )
        for i, col in enumerate(it, 1):
            frame = frame.with_columns(
                pl.when(pl.col(col) > pl.col(min_name))
                .then(i)
                .otherwise(pl.col(name))
                .alias(name),
                pl.when(pl.col(col) > pl.col(min_name))
                .then(pl.col(col))
                .otherwise(pl.col(min_name))
                .alias(min_name),
            )

        return frame.drop(min_name)
