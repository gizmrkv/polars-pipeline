from typing import Iterable, Literal, Tuple

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from polars import LazyFrame
from polars._typing import FrameType
from tqdm import tqdm

from polars_pipeline.transformer import Transformer
from polars_pipeline.utils import numerical_columns

from .utils import log_figure


class HistPlot(Transformer):
    def __init__(
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
    ):
        self.num_set = list(num_set) if num_set else None
        self.hue = hue
        self.stat = stat
        self.cumulative = cumulative
        self.common_bins = common_bins
        self.multiple = multiple
        self.element = element
        self.fill = fill
        self.kde = kde
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame not supported for plotting")

        num_set = self.num_set or numerical_columns(X)
        for num in tqdm(num_set, desc="Histogram"):
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.histplot(
                X,
                x=num,
                hue=self.hue,
                stat=self.stat,
                cumulative=self.cumulative,
                common_bins=self.common_bins,
                multiple=self.multiple,
                element=self.element,
                fill=self.fill,
                kde=self.kde,
                ax=ax,
            )

            title = f"Histogram of {num}"
            if self.hue:
                title += f" by {self.hue}"
            ax.set_title(title)

            log_figure(fig, title, log_dir=self.log_dir)
            fig.clear()
            plt.close(fig)

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X


class KDEPlot(Transformer):
    def __init__(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        multiple: Literal["layer", "stack", "fill"] = "layer",
        common_norm: bool = True,
        common_grid: bool = True,
        cumulative: bool = False,
        fill: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.num_set = list(num_set) if num_set else None
        self.hue = hue
        self.multiple = multiple
        self.common_norm = common_norm
        self.common_grid = common_grid
        self.cumulative = cumulative
        self.fill = fill
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise ValueError("LazyFrame not supported for plotting")

        num_set = self.num_set or numerical_columns(X)
        for num in tqdm(num_set, desc="KDE"):
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.kdeplot(
                X,
                x=num,
                hue=self.hue,
                multiple=self.multiple,
                common_norm=self.common_norm,
                common_grid=self.common_grid,
                cumulative=self.cumulative,
                fill=self.fill,
                ax=ax,
            )

            title = f"KDE of {num}"
            if self.hue:
                title += f" by {self.hue}"
            ax.set_title(title)

            log_figure(fig, title, log_dir=self.log_dir)
            fig.clear()
            plt.close(fig)

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X
