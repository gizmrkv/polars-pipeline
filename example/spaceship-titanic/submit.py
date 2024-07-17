from pathlib import Path

import polars as pl
from polars_pipeline import Pipeline, Transformer
from polars_pipeline.typing import FrameType

if __name__ == "__main__":
    pl.Config.set_tbl_cols(1000)
    pl.Config.set_tbl_rows(20)

    data_dir = Path("example/spaceship-titanic/data")
    train_df = pl.read_csv(data_dir / "train.csv")
    test_df = pl.read_csv(data_dir / "test.csv")
    sample_df = pl.read_csv(data_dir / "sample_submission.csv")

    all_df = pl.concat(
        [
            train_df,
            test_df.with_columns(pl.lit(None).alias("Transported")),
        ]
    )

    class Preprocess(Transformer):
        def fit_transform(self, X: FrameType, y: FrameType | None = None) -> FrameType:
            return self.transform(X).cast({"Transported": pl.UInt8})

        def transform(self, X: FrameType) -> FrameType:
            X = X.with_columns(
                (
                    pl.col("RoomService")
                    + pl.col("FoodCourt")
                    + pl.col("ShoppingMall")
                    + pl.col("Spa")
                    + pl.col("VRDeck")
                ).alias("TotalBill"),
                pl.col("Age").gt(12).alias("IsAdult"),
                pl.col("PassengerId").str.slice(0, 4).cast(pl.Int32).alias("Group"),
                pl.col("Cabin")
                .str.split("/")
                .list.get(1)
                .cast(pl.Int32)
                .alias("CabinNum"),
                pl.col("Cabin").str.split("/").list.get(2).alias("CabinSide"),
            ).cast(
                {
                    "HomePlanet": pl.Categorical,
                    "Destination": pl.Categorical,
                    "CabinSide": pl.Categorical,
                    "CryoSleep": pl.UInt8,
                    "VIP": pl.UInt8,
                }
            )
            X = (
                X.join(
                    X.group_by("Group").agg(pl.len().alias("GroupSize")),
                    on="Group",
                )
                .join(
                    X.group_by("Group").agg(
                        pl.sum("TotalBill").alias("GroupTotalBill")
                    ),
                    on="Group",
                )
                .with_columns(
                    (pl.col("TotalBill") / (pl.col("GroupTotalBill") + 1e-4)).alias(
                        "GroupBillRatio"
                    ),
                )
                .drop("PassengerId", "Name", "Cabin", "Group")
            )

            return X

    pipeline = (
        Pipeline(log_dir="./out")
        .pipe(Preprocess())
        .sort_columns()
        .plot.boxplot(hue="Transported")
        .plot.violinplot(hue="Transported")
        # .plot.histplot(hue="Transported")
        .plot.kdeplot(hue="Transported")
        .plot.scatterplot(hue="Transported")
        .plot.kde2dplot(hue="Transported")
        .plot.corr_heatmap()
        .plot.count_heatmap()
        .pre.label_encode()
    )
    pipeline.fit_transform(train_df)
