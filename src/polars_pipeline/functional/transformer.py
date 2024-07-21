from typing import Collection, Iterable, Literal, Mapping, Sequence

from polars import LazyFrame
from polars._typing import ColumnNameOrSelector, IntoExpr, PolarsDataType

from polars_pipeline.exception import LazyFrameNotSupportedError, NotFittedError
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType

from .horizontal import Horizontal


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
            raise NotFittedError(self.__class__.__name__)

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


class MeanHorizontal(Transformer):
    def __init__(self, columns: Sequence[str], *, name: str = "mean") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).mean(self.columns, name=self.name)


class SumHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "sum") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).sum(self.columns, name=self.name)


class ProdHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "prod") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).prod(self.columns, name=self.name)


class AllHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "all") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).all(self.columns, name=self.name)


class AnyHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "any") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).any(self.columns, name=self.name)


class MaxHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "max") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).max(self.columns, name=self.name)


class MinHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "min") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).min(self.columns, name=self.name)


class ArgmaxHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "argmax") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).argmax(self.columns, name=self.name)


class ArgminHorizontal(Transformer):
    def __init__(self, columns: Iterable[str], *, name: str = "argmin") -> None:
        self.columns = columns
        self.name = name

    def transform(self, X: FrameType) -> FrameType:
        return Horizontal(X).argmin(self.columns, name=self.name)


class Dummy(Transformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ):
        self.columns = columns
        self.separator = separator
        self.drop_first = drop_first

    def transform(self, X: FrameType) -> FrameType:
        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.transform.__name__
            )

        return X.to_dummies(
            self.columns, separator=self.separator, drop_first=self.drop_first
        )


class DropNulls(Transformer):
    def __init__(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ):
        self.subset = subset

    def transform(self, X: FrameType) -> FrameType:
        return X.drop_nulls(subset=self.subset)


class Cast(Transformer):
    def __init__(
        self,
        dtypes: (
            Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType]
            | PolarsDataType
        ),
        *,
        strict: bool = True,
    ):
        self.dtypes = dtypes
        self.strict = strict

    def transform(self, X: FrameType) -> FrameType:
        return X.cast(self.dtypes, strict=self.strict)
