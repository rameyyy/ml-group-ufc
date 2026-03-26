import polars as pl
from mapping import FIGHT_TYPE_MAP, WEIGHT_CLASS_MAP, STANCE_MAP, REFEREE_MAP


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
        pl.col("referee").replace(REFEREE_MAP).cast(pl.Int16).alias("referee_id"),
        # Fighter record
        pl.col("prior_cnt_f1").alias("f1_fight_count"),
        pl.col("prior_cnt_f2").alias("f2_fight_count"),
        # F1 physical
        pl.col("f1_height_in"),
        pl.col("f1_weight_lbs"),
        pl.col("f1_reach_in"),
        pl.col("f1_stance").replace(STANCE_MAP).cast(pl.Int8).alias("f1_stance_id"),
        (
            (pl.col("fight_date") - pl.col("f1_dob")).dt.total_days() / 365.25
        ).cast(pl.Float32).alias("f1_age"),
        # F1 record
        pl.col("f1_win"),
        pl.col("f1_loss"),
        (
            pl.col("f1_win").cast(pl.Float32) / (pl.col("f1_win") + pl.col("f1_loss")).cast(pl.Float32)
        ).alias("f1_win_rate"),
        # F1 UFC stats page aggregates (potential leakage — see FEATURES.md)
        pl.col("f1_slpm"),
        pl.col("f1_str_acc"),
        pl.col("f1_sapm"),
        pl.col("f1_str_def"),
        pl.col("f1_td_avg"),
        pl.col("f1_td_acc"),
        pl.col("f1_td_def"),
        pl.col("f1_sub_avg"),
        # F2 physical
        pl.col("f2_height_in"),
        pl.col("f2_weight_lbs"),
        pl.col("f2_reach_in"),
        pl.col("f2_stance").replace(STANCE_MAP).cast(pl.Int8).alias("f2_stance_id"),
        (
            (pl.col("fight_date") - pl.col("f2_dob")).dt.total_days() / 365.25
        ).cast(pl.Float32).alias("f2_age"),
        # F2 record
        pl.col("f2_win"),
        pl.col("f2_loss"),
        (
            pl.col("f2_win").cast(pl.Float32) / (pl.col("f2_win") + pl.col("f2_loss")).cast(pl.Float32)
        ).alias("f2_win_rate"),
        # F2 UFC stats page aggregates (potential leakage — see FEATURES.md)
        pl.col("f2_slpm"),
        pl.col("f2_str_acc"),
        pl.col("f2_sapm"),
        pl.col("f2_str_def"),
        pl.col("f2_td_avg"),
        pl.col("f2_td_acc"),
        pl.col("f2_td_def"),
        pl.col("f2_sub_avg"),
    ]
