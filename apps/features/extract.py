import polars as pl
from pathlib import Path

def get_fightsnapshots_df() -> pl.DataFrame:
    file = Path(__file__).parent.parent.parent / "data" / "fight_snapshots.parquet"
    df = pl.read_parquet(file)
    return df

def unnest_raw_df(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.with_columns(pl.col("fight_id").alias("root_fight_id"))

    fights_df = df.drop(["prior_f1", "prior_f2", "fight_id"])

    def extract_prior_fights(df, col, role):
        return (
            df.select(["root_fight_id", col])
            .explode(col)
            .unnest(col)
            .rename({"fight_id": "prior_fight_id"})
            .with_columns(pl.lit(role).alias("fighter_role"))
            .drop("rounds")
        )

    prior_fights_df = pl.concat([
        extract_prior_fights(df, "prior_f1", "f1"),
        extract_prior_fights(df, "prior_f2", "f2"),
    ])

    def extract_prior_rounds(df, col, role):
        return (
            df.select(["root_fight_id", col])
            .explode(col)
            .unnest(col)
            .rename({"fight_id": "prior_fight_id"})
            .with_columns(pl.lit(role).alias("fighter_role"))
            .select(["root_fight_id", "prior_fight_id", "fighter_role", "rounds"])
            .explode("rounds")
            .unnest("rounds")
        )

    prior_rounds_df = pl.concat([
        extract_prior_rounds(df, "prior_f1", "f1"),
        extract_prior_rounds(df, "prior_f2", "f2"),
    ])

    return fights_df, prior_fights_df, prior_rounds_df
