from typing import Iterable, Literal

from polars import LazyFrame
from polars._typing import ColumnNameOrSelector, IntoExpr

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType


class Select(Transformer):
    def __init__(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr):
        self.exprs = exprs
        self.named_exprs = named_exprs

    def transform(self, X: FrameType) -> FrameType:
        return X.select(*self.exprs, **self.named_exprs)


class WithColumns(Transformer):
    def __init__(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr):
        self.exprs = exprs
        self.named_exprs = named_exprs

    def transform(self, X: FrameType) -> FrameType:
        return X.with_columns(*self.exprs, **self.named_exprs)


class Drop(Transformer):
    def __init__(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ):
        self.columns = columns
        self.strict = strict

    def transform(self, X: FrameType) -> FrameType:
        return X.drop(*self.columns, strict=self.strict)


class SortColumns(Transformer):
    def __init__(
        self, by: Literal["dtype", "name"] = "dtype", descending: bool = False
    ):
        self.by = by
        self.descending = descending

    def transform(self, X: FrameType) -> FrameType:
        if isinstance(X, LazyFrame):
            raise NotImplementedError("SortColumns does not support LazyFrame")

        sorted_columns = sorted(
            [{"name": k, "dtype": str(v)} for k, v in X.schema.items()],
            key=lambda x: x[self.by],
            reverse=self.descending,
        )
        return X.select([col["name"] for col in sorted_columns])


class Display(Transformer):
    def transform(self, X: FrameType) -> FrameType:
        try:
            from IPython.display import display  # type: ignore
        except ImportError:
            return X

        display(X)
        return X
