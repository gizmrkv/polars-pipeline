from typing import Iterable, Tuple

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from polars import LazyFrame
from tqdm import tqdm

from polars_pipeline.exception import LazyFrameNotSupportedError
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType
from polars_pipeline.utils import categorical_columns, numerical_columns

from .utils import log_figure


class BoxPlot(Transformer):
    def __init__(
        self,
        *,
        num_set: Iterable[str] | None = None,
        cat_set: Iterable[str] | None = None,
        hue: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.num_set = list(num_set) if num_set else None
        self.cat_set = list(cat_set) if cat_set else None
        self.hue = hue
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        num_set = self.num_set or numerical_columns(X)
        cat_set = self.cat_set or categorical_columns(X)
        if self.hue:
            cat_set = [cat for cat in cat_set if cat != self.hue]

        total = len(num_set) * len(cat_set)
        pbar = tqdm(total=total, desc="Boxplot")
        for num in num_set:
            for cat in cat_set:
                fig, ax = plt.subplots(figsize=self.figsize)
                sns.boxplot(X, x=cat, y=num, hue=self.hue, ax=ax)
                title = f"Boxplot of {num} by {cat}"
                if self.hue:
                    title += f" and {self.hue}"
                ax.set_title(title)

                log_figure(fig, title, log_dir=self.log_dir)
                fig.clear()
                plt.close(fig)

                pbar.update()

        pbar.close()

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X


class ViolinPlot(Transformer):
    def __init__(
        self,
        *,
        num_set: Iterable[str] | None = None,
        cat_set: Iterable[str] | None = None,
        hue: str | None = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.num_set = list(num_set) if num_set else None
        self.cat_set = list(cat_set) if cat_set else None
        self.hue = hue
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        if y:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        num_set = self.num_set or numerical_columns(X)
        cat_set = self.cat_set or categorical_columns(X)
        if self.hue:
            cat_set = [cat for cat in cat_set if cat != self.hue]

        total = len(num_set) * len(cat_set)
        pbar = tqdm(total=total, desc="Violinplot")
        for num in num_set:
            for cat in cat_set:
                fig, ax = plt.subplots(figsize=self.figsize)
                sns.violinplot(X, x=cat, y=num, hue=self.hue, ax=ax)
                title = f"Violinplot of {num} by {cat}"
                if self.hue:
                    title += f" and {self.hue}"
                ax.set_title(title)

                log_figure(fig, title, log_dir=self.log_dir)
                fig.clear()
                plt.close(fig)

                pbar.update()

        pbar.close()

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X
