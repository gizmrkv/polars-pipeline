import datetime
import uuid
from pathlib import Path
from typing import Collection, Iterable, List, Literal, Mapping, Self, Sequence

from polars._typing import ColumnNameOrSelector, IntoExpr, PolarsDataType

from polars_pipeline import functional as F
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType

from .model import ModelNameSpace
from .plot import PlotNameSpace
from .preprocessing import PreprocessingNameSpace


class Pipeline(Transformer):
    def __init__(self, *, log_dir: Path | str | None = None) -> None:
        self.transformers: List[Transformer] = []

        if log_dir:
            self.log_dir = Path(log_dir)

    def set_log_dir(self):
        if log_dir := self.log_dir:
            log_dir = Path(log_dir) / datetime.datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S_{uuid.uuid4()}"
            )
            zero_pad = len(str(len(self.transformers)))
            for i, transformer in enumerate(self.transformers):
                name = f"{i:0>{zero_pad}}_{transformer.__class__.__name__}"
                transformer.log_dir = log_dir / name

    def fit(self, X: FrameType, y: FrameType | None = None):
        self.fit_transform(X, y)

    def transform(self, X: FrameType) -> FrameType:
        self.set_log_dir()
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X: FrameType, y: FrameType | None = None) -> FrameType:
        self.set_log_dir()
        for transformer in self.transformers:
            X = transformer.fit_transform(X, y)
        return X

    def pipe(self, transformer: Transformer) -> Self:
        self.transformers.append(transformer)
        return self

    @property
    def pre(self) -> PreprocessingNameSpace:
        return PreprocessingNameSpace(self)

    @property
    def plot(self) -> PlotNameSpace:
        return PlotNameSpace(self)

    @property
    def model(self) -> ModelNameSpace:
        return ModelNameSpace(self)

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(F.Select(*exprs, **named_exprs))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(F.WithColumns(*exprs, **named_exprs))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(F.Drop(*columns, strict=strict))

    def sort_columns(
        self, by: Literal["dtype", "name"] = "dtype", descending: bool = False
    ) -> Self:
        return self.pipe(F.SortColumns(by=by, descending=descending))

    def display(self) -> Self:
        return self.pipe(F.Display())

    def mean_horizontal(self, columns: Sequence[str], *, name: str = "mean") -> Self:
        return self.pipe(F.MeanHorizontal(columns, name=name))

    def sum_horizontal(self, columns: Iterable[str], *, name: str = "sum") -> Self:
        return self.pipe(F.SumHorizontal(columns, name=name))

    def prod_horizontal(self, columns: Iterable[str], *, name: str = "prod") -> Self:
        return self.pipe(F.ProdHorizontal(columns, name=name))

    def all_horizontal(self, columns: Iterable[str], *, name: str = "all") -> Self:
        return self.pipe(F.AllHorizontal(columns, name=name))

    def any_horizontal(self, columns: Iterable[str], *, name: str = "any") -> Self:
        return self.pipe(F.AnyHorizontal(columns, name=name))

    def max_horizontal(self, columns: Iterable[str], *, name: str = "max") -> Self:
        return self.pipe(F.MaxHorizontal(columns, name=name))

    def min_horizontal(self, columns: Iterable[str], *, name: str = "min") -> Self:
        return self.pipe(F.MinHorizontal(columns, name=name))

    def argmax_horizontal(
        self, columns: Iterable[str], *, name: str = "argmax"
    ) -> Self:
        return self.pipe(F.ArgmaxHorizontal(columns, name=name))

    def argmin_horizontal(
        self, columns: Iterable[str], *, name: str = "argmin"
    ) -> Self:
        return self.pipe(F.ArgminHorizontal(columns, name=name))

    def dummy(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ) -> Self:
        return self.pipe(F.Dummy(columns, separator=separator, drop_first=drop_first))

    def drop_nulls(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(F.DropNulls(columns))

    def cast(
        self,
        dtypes: (
            Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType]
            | PolarsDataType
        ),
        *,
        strict: bool = True,
    ) -> Self:
        return self.pipe(F.Cast(dtypes, strict=strict))
