import polars as pl
from mapping import FIGHT_TYPE_MAP, WEIGHT_CLASS_MAP, STANCE_MAP


def fights_select_exprs() -> list:
    return [
        # Metadata
        pl.col("root_fight_id").alias("meta_root_fight_id"),
        pl.col("fighter1_id").alias("meta_f1_id"),
        pl.col("fighter2_id").alias("meta_f2_id"),
        pl.col("winner_id").alias("meta_winner_id"),
        pl.col("loser_id").alias("meta_loser_id"),
        pl.col("fight_date").alias("meta_fight_date"),
        (
            pl.col("end_time").str.split(":").list.get(0).cast(pl.Int16) * 60
            + pl.col("end_time").str.split(":").list.get(1).cast(pl.Int16)
        ).alias("meta_end_time"),
        pl.col("method").alias("meta_method"),
        pl.col("fight_type").alias("meta_fight_type"),
        # Fight context
        pl.col("fight_format").cast(pl.Int16).alias("fight_format"),
        pl.col("fight_type").replace(FIGHT_TYPE_MAP).cast(pl.Int8).alias("fight_type_id"),
        pl.col("weight_class").replace(WEIGHT_CLASS_MAP).cast(pl.Int8).alias("weight_class_id"),
        # Fighter record (absolute counts kept individually — symmetry not assumed)
        pl.col("prior_cnt_f1").alias("f1_fight_count"),
        pl.col("prior_cnt_f2").alias("f2_fight_count"),
        # Stance (categorical — kept per-fighter)
        pl.col("f1_stance").replace(STANCE_MAP).cast(pl.Int8).alias("f1_stance_id"),
        pl.col("f2_stance").replace(STANCE_MAP).cast(pl.Int8).alias("f2_stance_id"),
        # Physical differentials (f1 - f2)
        (pl.col("f1_height_in") - pl.col("f2_height_in")).alias("height_diff"),
        (pl.col("f1_reach_in") - pl.col("f2_reach_in")).alias("reach_diff"),
        (
            (pl.col("fight_date") - pl.col("f1_dob")).dt.total_days() / 365.25
        ).cast(pl.Float32).alias("f1_age"),
        (
            (pl.col("fight_date") - pl.col("f2_dob")).dt.total_days() / 365.25
        ).cast(pl.Float32).alias("f2_age"),
        (
            (pl.col("f2_dob") - pl.col("f1_dob")).dt.total_days() / 365.25
        ).cast(pl.Float32).alias("age_diff"),
        # Win rate differentials
        (
            pl.col("f1_win").cast(pl.Float32) / (pl.col("f1_win") + pl.col("f1_loss")).cast(pl.Float32)
            - pl.col("f2_win").cast(pl.Float32) / (pl.col("f2_win") + pl.col("f2_loss")).cast(pl.Float32)
        ).alias("win_rate_diff"),
        # Southpaw advantage: 1 if f1=southpaw & f2=orthodox, -1 if reversed, 0 otherwise
        pl.when(
            (pl.col("f1_stance") == "Southpaw") & (pl.col("f2_stance") == "Orthodox")
        ).then(pl.lit(1))
        .when(
            (pl.col("f1_stance") == "Orthodox") & (pl.col("f2_stance") == "Southpaw")
        ).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("southpaw_advantage"),
    ]
