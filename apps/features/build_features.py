import polars as pl

METHOD_MAP = {
    "d_unan": 0,
    "d_maj": 1,
    "d_split": 2,
    "kotko": 3,
    "sub": 4,
    "unknown": 5,
    None: 6,
}

FIGHT_TYPE_MAP = {
    "main": 0,
    "title": 1,
    None: 2,
}

class FightFeatures:
    def __init__(
        self,
        fights_df: pl.DataFrame,
        prior_fights_df: pl.DataFrame,
        prior_rounds_df: pl.DataFrame,
    ):
        self.fights = fights_df
        self.prior_fights = prior_fights_df
        self.prior_rounds = prior_rounds_df
        self.final_df = pl.DataFrame

    def root_fights_meta_df(self):
        self.final_df = self.fights.select(
            pl.col("root_fight_id").alias("meta_root_fight_id"),
            pl.col("fighter1_id").alias("meta_f1_id"),
            pl.col("fighter2_id").alias("meta_f2_id"),
            pl.col("winner_id").alias("meta_winner_id"),
            pl.col("loser_id").alias("meta_loser_id"),
            pl.col("fight_date").alias("meta_fight_date"),
            pl.col("method").alias("meta_method"),
            pl.col("fight_type").alias("meta_fight_type"),
            pl.col("fight_format").cast(pl.Int16).alias("fight_format"),
            pl.col("fight_type").replace(FIGHT_TYPE_MAP).cast(pl.Int8).alias("fight_type_id"),
        )