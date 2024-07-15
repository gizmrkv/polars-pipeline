from typing import TYPE_CHECKING, Sequence, Tuple

if TYPE_CHECKING:
    from polars_pipeline import Pipeline

from polars_pipeline.preprocessing import (
    Binarizer,
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class PreprocessingNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def binarize(
        self, columns: str | Sequence[str] | None = None, *, threshold: float = 0.5
    ) -> "Pipeline":
        return self.pipeline.pipe(Binarizer(columns, threshold=threshold))

    def label_encode(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        maintain_order: bool = False,
    ) -> "Pipeline":
        return self.pipeline.pipe(LabelEncoder(columns, maintain_order=maintain_order))

    def min_max_scale(self, columns: str | Sequence[str]) -> "Pipeline":
        return self.pipeline.pipe(MinMaxScaler(columns))

    def robust_scale(
        self,
        columns: str | Sequence[str],
        *,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
    ) -> "Pipeline":
        return self.pipeline.pipe(RobustScaler(columns, quantile_range=quantile_range))

    def standard_scale(self, columns: str | Sequence[str]) -> "Pipeline":
        return self.pipeline.pipe(StandardScaler(columns))
