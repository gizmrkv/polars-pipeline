from typing import Iterable, Tuple

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from polars import LazyFrame
from tqdm import tqdm

from polars_pipeline.exception import LazyFrameNotSupportedError
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType
from polars_pipeline.utils import numerical_columns

from .utils import log_figure


class ScatterPlot(Transformer):
    def __init__(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        size: str | None = None,
        style: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.num_set = list(num_set) if num_set else None
        self.hue = hue
        self.size = size
        self.style = style
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        num_set = self.num_set or numerical_columns(X)
        total = len(num_set) * (len(num_set) - 1) // 2
        pbar = tqdm(total=total, desc="Scatter")
        for i, num1 in enumerate(num_set):
            for num2 in num_set[i + 1 :]:
                df = X.drop_nulls(
                    [num1, num2]
                    + ([self.hue] if self.hue else [])
                    + ([self.size] if self.size else [])
                    + ([self.style] if self.style else [])
                )
                fig, ax = plt.subplots(figsize=self.figsize)
                sns.scatterplot(
                    df,
                    x=num1,
                    y=num2,
                    hue=self.hue,
                    size=self.size,
                    style=self.style,
                    ax=ax,
                )

                title = f"Scatter of {num1} and {num2}"
                if self.hue:
                    title += f" by {self.hue}"
                ax.set_title(title)

                log_figure(fig, title, log_dir=self.log_dir)
                fig.clear()
                plt.close(fig)

                pbar.update()

        pbar.close()

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X


class KDE2dPlot(Transformer):
    def __init__(
        self,
        *,
        num_set: Iterable[str] | None = None,
        hue: str | None = None,
        fill: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.num_set = list(num_set) if num_set else None
        self.hue = hue
        self.fill = fill
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        num_set = self.num_set or numerical_columns(X)
        total = len(num_set) * (len(num_set) - 1) // 2
        pbar = tqdm(total=total, desc="KDE 2D")
        for i, num1 in enumerate(num_set):
            for num2 in num_set[i + 1 :]:
                df = X.drop_nulls([num1, num2] + ([self.hue] if self.hue else []))

                fig, ax = plt.subplots(figsize=self.figsize)
                sns.kdeplot(df, x=num1, y=num2, hue=self.hue, fill=self.fill, ax=ax)

                title = f"KDE 2D plot of {num1} and {num2}"
                if self.hue:
                    title += f" by {self.hue}"
                ax.set_title(title)

                log_figure(fig, title, log_dir=self.log_dir)
                fig.clear()
                plt.close(fig)

                pbar.update()

        pbar.close()

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X
