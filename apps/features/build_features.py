import polars as pl
from fights_exprs import fights_select_exprs
from prior_fights_exprs import build_duration_features, build_format_experience, build_years_since_last_fight, build_activity, build_method_counts, build_weight_class_fight_count, split_by_role


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

    def extract_fights_features(self):
        self.final_df = self.fights.select(fights_select_exprs())

    def extract_prior_fights_features(self):
        root_dates = self.fights.select(["root_fight_id", pl.col("fight_date").alias("root_fight_date")])

        days_since = build_years_since_last_fight(self.prior_fights, root_dates)
        activity = build_activity(self.prior_fights, root_dates)
        method_counts = build_method_counts(self.prior_fights)
        weight_class_counts = build_weight_class_fight_count(self.prior_fights, self.fights)
        duration = build_duration_features(self.prior_fights)
        format_exp = build_format_experience(self.prior_fights)

        self.final_df = (
            self.final_df
            .join(split_by_role(days_since, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(days_since, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(activity, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(activity, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(method_counts, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(method_counts, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(weight_class_counts, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(weight_class_counts, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(duration, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(duration, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(format_exp, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(format_exp, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
        )
