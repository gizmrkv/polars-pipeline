import uuid
from typing import Iterable, Tuple

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.typing import ColorType
from polars import LazyFrame
from tqdm import tqdm

from polars_pipeline.exception import LazyFrameNotSupportedError
from polars_pipeline.transformer import Transformer
from polars_pipeline.typing import FrameType
from polars_pipeline.utils import categorical_columns, numerical_columns

from .utils import log_figure


class CorrelationHeatmap(Transformer):
    def __init__(
        self,
        *,
        cmap: str | list[ColorType] | Colormap = "coolwarm",
        annot: bool = False,
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.cmap = cmap
        self.annot = annot
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        log_dir = self.log_dir
        if log_dir is None:
            return

        if y is not None:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        X = X.select(numerical_columns(X)).drop_nulls()
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            X.corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap=self.cmap,
            center=0.0,
            annot=self.annot,
            fmt=".2f",
            cbar=True,
            square=True,
            xticklabels=X.columns,
            yticklabels=X.columns,
            ax=ax,
        )
        title = "Correlation Heatmap"
        ax.set_title(title)

        log_figure(fig, title, log_dir)
        fig.clear()
        plt.close(fig)

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X


class CountHeatmap(Transformer):
    def __init__(
        self,
        *,
        cat_set: Iterable[str] | None = None,
        sort_by_index: bool = True,
        sort_columns: bool = True,
        cmap: str | list[ColorType] | Colormap = "viridis",
        figsize: Tuple[int, int] = (10, 8),
    ):
        self.cat_set = list(cat_set) if cat_set else None
        self.sort_by_index = sort_by_index
        self.sort_columns = sort_columns
        self.cmap = cmap
        self.figsize = figsize

    def log_figures(self, X: FrameType, y: FrameType | None = None):
        log_dir = self.log_dir
        if log_dir is None:
            return

        if y is not None:
            X = pl.concat([X, y], how="horizontal")

        if isinstance(X, LazyFrame):
            raise LazyFrameNotSupportedError(
                self.__class__.__name__, self.log_figures.__name__
            )

        cat_set = self.cat_set or categorical_columns(X)
        total = len(cat_set) * (len(cat_set) - 1) // 2
        pbar = tqdm(total=total, desc="Count Heatmap")
        for i, cat1 in enumerate(cat_set):
            for cat2 in cat_set[i + 1 :]:
                count_name = str(uuid.uuid4())
                count_df = (
                    X.select(cat1, cat2)
                    .group_by([cat1, cat2])
                    .agg(pl.len().alias(count_name))
                    .pivot(
                        cat2,
                        index=cat1,
                        values=count_name,
                        sort_columns=self.sort_columns,
                    )
                    .fill_null(0)
                )
                if self.sort_by_index:
                    count_df = count_df.sort(cat1)

                cat1_labels = [
                    ("null" if label is None else label)
                    for label in count_df.get_column(cat1).to_list()
                ]
                cat2_labels = count_df.columns[1:]

                fig, ax = plt.subplots(figsize=self.figsize)
                sns.heatmap(
                    count_df.drop(cat1),
                    vmin=0,
                    cmap=self.cmap,
                    robust=True,
                    annot=True,
                    fmt="",
                    xticklabels=cat2_labels,
                    yticklabels=cat1_labels,
                    ax=ax,
                )
                title = f"Count of {cat1} vs {cat2}"
                ax.set_title(title)
                ax.set_ylabel(cat1)
                ax.set_xlabel(cat2)

                log_figure(fig, title, log_dir)
                fig.clear()
                plt.close(fig)

                pbar.update()

        pbar.close()

    def transform(self, X: FrameType) -> FrameType:
        self.log_figures(X)
        return X
