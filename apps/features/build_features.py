import polars as pl
from fights_exprs import fights_select_exprs
from prior_fights_exprs import build_duration_features, build_format_experience, build_years_since_last_fight, build_activity, build_method_counts, build_weight_class_fight_count, build_striking_stats, build_advanced_stats, build_round_features, split_by_role


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
        striking = build_striking_stats(self.prior_fights)
        advanced = build_advanced_stats(self.prior_fights)
        rounds = build_round_features(self.prior_rounds, self.prior_fights)

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
            .join(split_by_role(striking, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(striking, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(advanced, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(advanced, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(rounds, "f1", "f1"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
            .join(split_by_role(rounds, "f2", "f2"), left_on="meta_root_fight_id", right_on="root_fight_id", how="left")
        )
        self._compute_striking_diffs()
        self._compute_advanced_diffs()
        self._compute_round_diffs()

    def _compute_striking_diffs(self):
        stats = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
        windows = ["last_fight", "last_3", ""]  # "" = career
        trend_stats = [f"{s}_trend" for s in stats]

        diff_exprs = []
        drop_cols = []
        for w in windows:
            for s in stats:
                f1 = f"f1_{w}_{s}".strip("_").replace("f1__", "f1_")
                f2 = f"f2_{w}_{s}".strip("_").replace("f2__", "f2_")
                diff = f"{w}_{s}_diff".strip("_")
                diff_exprs.append((pl.col(f1) - pl.col(f2)).alias(diff))
                drop_cols.extend([f1, f2])
        for s in trend_stats:
            diff_exprs.append((pl.col(f"f1_{s}") - pl.col(f"f2_{s}")).alias(f"{s}_diff"))
            drop_cols.extend([f"f1_{s}", f"f2_{s}"])

        self.final_df = self.final_df.with_columns(diff_exprs).drop(drop_cols)

    def _compute_round_diffs(self):
        career_stats = [
            "r1_sig_per_fight", "r1_kd_rate",
            "slpm_pace_ratio", "sapm_pace_ratio",
            "str_acc_degradation", "body_escalation",
            "late_ctrl_per_fight", "late_td_acc", "late_sub_per_round",
            "rd_dom_rate", "last_rd_str_diff",
            "post_kd_response",
        ]
        last_3_stats = [
            "last_3_r1_sig_per_fight",
            "last_3_slpm_pace_ratio", "last_3_sapm_pace_ratio",
            "last_3_str_acc_degradation",
            "last_3_rd_dom_rate", "last_3_last_rd_str_diff",
            "last_3_post_kd_response",
        ]

        diff_exprs = []
        drop_cols = []
        for s in career_stats + last_3_stats:
            diff_exprs.append((pl.col(f"f1_{s}") - pl.col(f"f2_{s}")).alias(f"{s}_diff"))
            drop_cols.extend([f"f1_{s}", f"f2_{s}"])

        self.final_df = self.final_df.with_columns(diff_exprs).drop(drop_cols)

    def _compute_advanced_diffs(self):
        # (stat, windows) — empty string = career only window
        stat_windows = [
            ("kd_rate",            ["last_fight", "last_3", ""]),
            ("net_ctrl_pct",       ["last_3", ""]),
            ("sig_to_total_ratio", ["last_3", ""]),
            ("body_str_pct",       [""]),
            ("leg_str_pct",        [""]),
            ("clinch_str_pct",     [""]),
            ("ground_str_pct",     [""]),
            ("gnp_rate",           [""]),
            ("chin_score",         [""]),
            ("reversal_rate",      [""]),
            ("def_sub_exposure",   [""]),
        ]

        diff_exprs = []
        drop_cols = []
        for stat, windows in stat_windows:
            for w in windows:
                f1 = f"f1_{w}_{stat}".strip("_").replace("f1__", "f1_")
                f2 = f"f2_{w}_{stat}".strip("_").replace("f2__", "f2_")
                diff = f"{w}_{stat}_diff".strip("_")
                diff_exprs.append((pl.col(f1) - pl.col(f2)).alias(diff))
                drop_cols.extend([f1, f2])

        self.final_df = self.final_df.with_columns(diff_exprs).drop(drop_cols)
