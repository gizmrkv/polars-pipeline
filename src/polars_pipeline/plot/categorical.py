from typing import Iterable, Tuple

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from polars import LazyFrame
from polars._typing import FrameType
from tqdm import tqdm

from polars_pipeline.transformer import Transformer
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
            raise ValueError("LazyFrame not supported for plotting")

        num_set = self.num_set or numerical_columns(X)
        cat_set = self.cat_set or categorical_columns(X)
        total = len(num_set) * (len(cat_set) - (1 if self.hue else 0))
        pbar = tqdm(total=total, desc="Boxplot")

        for num in num_set:
            for cat in cat_set:
                if cat == self.hue:
                    continue

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

    def fit(self, X: FrameType, y: FrameType | None = None):
        self.log_figures(X, y)

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
            raise ValueError("LazyFrame not supported for plotting")

        num_set = self.num_set or numerical_columns(X)
        cat_set = self.cat_set or categorical_columns(X)
        total = len(num_set) * (len(cat_set) - (1 if self.hue else 0))
        pbar = tqdm(total=total, desc="Violinplot")

        for num in num_set:
            for cat in cat_set:
                if cat == self.hue:
                    continue

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

    def fit(self, X: FrameType, y: FrameType | None = None):
        self.log_figures(X, y)

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X
