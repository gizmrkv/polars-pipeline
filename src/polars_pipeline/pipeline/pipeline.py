import datetime
import uuid
from pathlib import Path
from typing import Iterable, List, Literal, Self

from polars._typing import ColumnNameOrSelector, IntoExpr

from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType

from .functional import Display, Drop, Select, SortColumns, WithColumns
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

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(Select(*exprs, **named_exprs))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(WithColumns(*exprs, **named_exprs))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(Drop(*columns, strict=strict))

    def sort_columns(
        self, by: Literal["dtype", "name"] = "dtype", descending: bool = False
    ) -> Self:
        return self.pipe(SortColumns(by=by, descending=descending))

    def display(self) -> Self:
        return self.pipe(Display())
