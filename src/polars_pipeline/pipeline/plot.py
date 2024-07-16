from typing import TYPE_CHECKING, Iterable, Literal, Tuple

from matplotlib.colors import Colormap
from matplotlib.typing import ColorType

if TYPE_CHECKING:
    from polars_pipeline import Pipeline

from polars_pipeline.plot import (
    BoxPlot,
    CorrelationHeatmap,
    CountHeatmap,
    HistPlot,
    KDE2dPlot,
    KDEPlot,
    ScatterPlot,
    ViolinPlot,
)


class PlotNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def boxplot(
        self,
        *,
        num_set: Iterable[str] | None = None,
        cat_set: Iterable[str] | None = None,
        hue: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            BoxPlot(num_set=num_set, cat_set=cat_set, hue=hue, figsize=figsize)
        )

    def violinplot(
        self,
        *,
        num_set: Iterable[str] | None = None,
        cat_set: Iterable[str] | None = None,
        hue: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ViolinPlot(num_set=num_set, cat_set=cat_set, hue=hue, figsize=figsize)
        )

    def histplot(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        stat: Literal[
            "count", "frequency", "probability", "percent", "density"
        ] = "count",
        cumulative: bool = False,
        common_bins: bool = True,
        multiple: Literal["layer", "dodge", "stack", "fill"] = "layer",
        element: Literal["bars", "step", "poly"] = "bars",
        fill: bool = True,
        kde: bool = False,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            HistPlot(
                num_set=num_set,
                hue=hue,
                stat=stat,
                cumulative=cumulative,
                common_bins=common_bins,
                multiple=multiple,
                element=element,
                fill=fill,
                kde=kde,
                figsize=figsize,
            )
        )

    def kdeplot(
        self,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        multiple: Literal["layer", "stack", "fill"] = "layer",
        common_norm: bool = True,
        common_grid: bool = True,
        cumulative: bool = False,
        fill: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            KDEPlot(
                num_set=num_set,
                hue=hue,
                multiple=multiple,
                common_norm=common_norm,
                common_grid=common_grid,
                cumulative=cumulative,
                fill=fill,
                figsize=figsize,
            )
        )

    def corr_heatmap(
        self,
        *,
        cmap: str | list[ColorType] | Colormap = "coolwarm",
        annot: bool = False,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CorrelationHeatmap(cmap=cmap, annot=annot, figsize=figsize)
        )

    def count_heatmap(
        self,
        *,
        cat_set: Iterable[str] | None = None,
        sort_by_index: bool = True,
        sort_columns: bool = True,
        cmap: str | list[ColorType] | Colormap = "viridis",
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CountHeatmap(
                cat_set=cat_set,
                sort_by_index=sort_by_index,
                sort_columns=sort_columns,
                cmap=cmap,
                figsize=figsize,
            )
        )

    def scatterplot(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        size: str | None = None,
        style: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ScatterPlot(
                num_set=num_set,
                hue=hue,
                size=size,
                style=style,
                figsize=figsize,
            )
        )

    def kde2dplot(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        fill: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ) -> "Pipeline":
        return self.pipeline.pipe(
            KDE2dPlot(num_set=num_set, hue=hue, fill=fill, figsize=figsize)
        )
