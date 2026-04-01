import polars as pl


def build_duration_features(prior_fights: pl.DataFrame) -> pl.DataFrame:
    end_time_s = (
        pl.col("end_time").str.split(":").list.get(0).cast(pl.Int32) * 60
        + pl.col("end_time").str.split(":").list.get(1).cast(pl.Int32)
    )

    pf = prior_fights.with_columns(end_time_s.alias("end_time_s"))

    return (
        pf
        .group_by(["root_fight_id", "fighter_role"])
        .agg([
            pl.col("end_time_s").sort_by("fight_date", descending=True).first().alias("last_fight_end_time_s"),
            pl.col("end_time_s").sort_by("fight_date", descending=True).head(3).mean().alias("last_3_avg_end_time_s"),
            pl.col("end_time_s").mean().alias("avg_end_time_s"),
            pl.col("end_time_s").sum().alias("total_time_fought_s"),
        ])
    )


def build_format_experience(prior_fights: pl.DataFrame) -> pl.DataFrame:
    win = pl.col("result") == "win"
    loss = pl.col("result") == "loss"
    r3 = pl.col("fight_format") == 3
    r5 = pl.col("fight_format") == 5

    return (
        prior_fights
        .group_by(["root_fight_id", "fighter_role"])
        .agg([
            (r3 & win).sum().alias("3rd_wins"),
            (r3 & loss).sum().alias("3rd_losses"),
            r5.sum().alias("5rd_fights"),
            (r5 & win).sum().alias("5rd_wins"),
            (r5 & loss).sum().alias("5rd_losses"),
        ])
    )


def build_years_since_last_fight(prior_fights: pl.DataFrame, root_dates: pl.DataFrame) -> pl.DataFrame:
    return (
        prior_fights
        .group_by(["root_fight_id", "fighter_role"])
        .agg(pl.col("fight_date").max().alias("last_fight_date"))
        .join(root_dates, on="root_fight_id")
        .with_columns(
            ((pl.col("root_fight_date") - pl.col("last_fight_date")).dt.total_days() / 365.25)
            .cast(pl.Float32)
            .alias("years_since_last_fight")
        )
        .select(["root_fight_id", "fighter_role", "years_since_last_fight"])
    )


def build_activity(prior_fights: pl.DataFrame, root_dates: pl.DataFrame) -> pl.DataFrame:
    return (
        prior_fights
        .join(root_dates, on="root_fight_id")
        .group_by(["root_fight_id", "fighter_role"])
        .agg([
            pl.col("fight_date").min().alias("first_fight_date"),
            pl.col("fight_date").count().alias("total_fights"),
            pl.col("root_fight_date").first(),
            pl.col("fight_date").filter(
                pl.col("fight_date") >= (pl.col("root_fight_date") - pl.duration(days=365))
            ).count().alias("fights_this_year"),
            pl.col("fight_date").filter(
                pl.col("fight_date") >= (pl.col("root_fight_date") - pl.duration(days=1095))
            ).count().alias("fights_last_3yrs"),
        ])
        .with_columns(
            (
                pl.col("total_fights").cast(pl.Float32) /
                ((pl.col("root_fight_date") - pl.col("first_fight_date")).dt.total_days() / 365.25)
            ).alias("avg_fights_per_year")
        )
        .select(["root_fight_id", "fighter_role", "avg_fights_per_year", "fights_this_year", "fights_last_3yrs"])
    )


def build_method_counts(prior_fights: pl.DataFrame) -> pl.DataFrame:
    win = pl.col("result") == "win"
    loss = pl.col("result") == "loss"
    ko = pl.col("method") == "kotko"
    sub = pl.col("method") == "sub"
    dec = pl.col("method").is_in(["d_unan", "d_maj", "d_split"])

    return (
        prior_fights
        .group_by(["root_fight_id", "fighter_role"])
        .agg([
            (win & ko).sum().alias("ko_wins"),
            (win & sub).sum().alias("sub_wins"),
            (win & dec).sum().alias("dec_wins"),
            (loss & ko).sum().alias("ko_losses"),
            (loss & sub).sum().alias("sub_losses"),
            (loss & dec).sum().alias("dec_losses"),
            win.sum().alias("total_wins"),
            loss.sum().alias("total_losses"),
        ])
        .with_columns([
            (pl.col("ko_wins").cast(pl.Float32) / pl.col("total_wins").cast(pl.Float32)).fill_nan(0.0).alias("ko_win_rate"),
            (pl.col("sub_wins").cast(pl.Float32) / pl.col("total_wins").cast(pl.Float32)).fill_nan(0.0).alias("sub_win_rate"),
            (pl.col("ko_losses").cast(pl.Float32) / pl.col("total_losses").cast(pl.Float32)).fill_nan(0.0).alias("ko_loss_rate"),
            (pl.col("sub_losses").cast(pl.Float32) / pl.col("total_losses").cast(pl.Float32)).fill_nan(0.0).alias("sub_loss_rate"),
        ])
        .drop(["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses", "total_wins", "total_losses"])
    )


def build_striking_stats(prior_fights: pl.DataFrame) -> pl.DataFrame:
    end_time_s = (
        pl.col("end_time").str.split(":").list.get(0).cast(pl.Float32) * 60
        + pl.col("end_time").str.split(":").list.get(1).cast(pl.Float32)
    )

    pf = prior_fights.with_columns(end_time_s.alias("end_time_s"))

    # Raw numerators/denominators needed for each window
    agg_cols = [
        "sig_str_landed", "sig_str_attempts",
        "opp_sig_str_landed", "opp_sig_str_attempts",
        "td_landed", "td_attempts",
        "opp_td_landed", "opp_td_attempts",
        "sub_att", "end_time_s",
    ]

    last_fight_aggs = [
        pl.col(c).sort_by("fight_date", descending=True).first().alias(f"lf_{c}")
        for c in agg_cols
    ]
    last_3_aggs = [
        pl.col(c).sort_by("fight_date", descending=True).head(3).sum().alias(f"l3_{c}")
        for c in agg_cols
    ]
    career_aggs = [
        pl.col(c).sum().alias(f"ca_{c}")
        for c in agg_cols
    ]

    def stat_exprs(col_p: str, alias_p: str) -> list:
        mins = pl.col(f"{col_p}_end_time_s") / 60.0
        a = f"{alias_p}_" if alias_p else ""

        def safe(expr, alias):
            return (
                pl.when(expr.is_nan() | expr.is_infinite())
                .then(pl.lit(0.0))
                .otherwise(expr)
                .alias(alias)
            )

        return [
            safe(pl.col(f"{col_p}_sig_str_landed") / mins, f"{a}slpm"),
            safe(pl.col(f"{col_p}_sig_str_landed") / pl.col(f"{col_p}_sig_str_attempts"), f"{a}str_acc"),
            safe(pl.col(f"{col_p}_opp_sig_str_landed") / mins, f"{a}sapm"),
            safe(1 - pl.col(f"{col_p}_opp_sig_str_landed") / pl.col(f"{col_p}_opp_sig_str_attempts"), f"{a}str_def"),
            safe(pl.col(f"{col_p}_td_landed") / mins * 15, f"{a}td_avg"),
            safe(pl.col(f"{col_p}_td_landed") / pl.col(f"{col_p}_td_attempts"), f"{a}td_acc"),
            safe(1 - pl.col(f"{col_p}_opp_td_landed") / pl.col(f"{col_p}_opp_td_attempts"), f"{a}td_def"),
            safe(pl.col(f"{col_p}_sub_att") / mins * 15, f"{a}sub_avg"),
        ]

    stats = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]

    # Trend: last_3 / career — > 1 = improving, < 1 = declining
    # Clipped to [0, 3] to avoid runaway ratios on small samples; fill 0/0 = 1 (neutral)
    trend_exprs = [
        pl.when((pl.col(f"last_3_{s}") / pl.col(s)).is_nan() | (pl.col(f"last_3_{s}") / pl.col(s)).is_infinite())
        .then(pl.lit(1.0))
        .otherwise((pl.col(f"last_3_{s}") / pl.col(s)).clip(0.0, 3.0))
        .alias(f"{s}_trend")
        for s in stats
    ]

    output_cols = (
        ["root_fight_id", "fighter_role"]
        + [f"last_fight_{s}" for s in stats]
        + [f"last_3_{s}" for s in stats]
        + stats
        + [f"{s}_trend" for s in stats]
    )

    return (
        pf
        .group_by(["root_fight_id", "fighter_role"])
        .agg(last_fight_aggs + last_3_aggs + career_aggs)
        .with_columns(stat_exprs("lf", "last_fight") + stat_exprs("l3", "last_3") + stat_exprs("ca", ""))
        .with_columns(trend_exprs)
        .select(output_cols)
    )


def build_advanced_stats(prior_fights: pl.DataFrame) -> pl.DataFrame:
    end_time_s = (
        pl.col("end_time").str.split(":").list.get(0).cast(pl.Float32) * 60
        + pl.col("end_time").str.split(":").list.get(1).cast(pl.Float32)
    )

    pf = prior_fights.with_columns(end_time_s.alias("end_time_s"))

    raw_cols = [
        "kd", "sig_str_landed",
        "ctrl_time_s", "opp_ctrl_time_s", "end_time_s",
        "head_landed", "body_landed", "leg_landed",
        "distance_landed", "clinch_landed", "ground_landed",
        "opp_kd", "opp_sig_str_landed",
        "rev", "opp_sub_att", "opp_td_landed",
        "total_str_landed",
    ]

    last_fight_aggs = [
        pl.col(c).sort_by("fight_date", descending=True).first().alias(f"lf_{c}")
        for c in raw_cols
    ]
    last_3_aggs = [
        pl.col(c).sort_by("fight_date", descending=True).head(3).sum().alias(f"l3_{c}")
        for c in raw_cols
    ]
    career_aggs = [
        pl.col(c).sum().alias(f"ca_{c}")
        for c in raw_cols
    ]

    def derived(col_p: str, alias_p: str) -> list:
        a = f"{alias_p}_" if alias_p else ""
        ctrl_mins = pl.col(f"{col_p}_ctrl_time_s") / 60.0
        opp_ctrl_mins = pl.col(f"{col_p}_opp_ctrl_time_s") / 60.0
        def safe(expr, alias):
            return (
                pl.when(expr.is_nan() | expr.is_infinite())
                .then(pl.lit(0.0))
                .otherwise(expr)
                .alias(alias)
            )

        return [
            safe(pl.col(f"{col_p}_kd") / pl.col(f"{col_p}_sig_str_landed"), f"{a}kd_rate"),
            safe((pl.col(f"{col_p}_ctrl_time_s") - pl.col(f"{col_p}_opp_ctrl_time_s")) / pl.col(f"{col_p}_end_time_s"), f"{a}net_ctrl_pct"),
            safe(pl.col(f"{col_p}_head_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}head_str_pct"),
            safe(pl.col(f"{col_p}_body_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}body_str_pct"),
            safe(pl.col(f"{col_p}_leg_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}leg_str_pct"),
            safe(pl.col(f"{col_p}_distance_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}distance_str_pct"),
            safe(pl.col(f"{col_p}_clinch_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}clinch_str_pct"),
            safe(pl.col(f"{col_p}_ground_landed") / pl.col(f"{col_p}_sig_str_landed"), f"{a}ground_str_pct"),
            safe(pl.col(f"{col_p}_ground_landed") / ctrl_mins, f"{a}gnp_rate"),
            safe(pl.col(f"{col_p}_opp_kd") / pl.col(f"{col_p}_opp_sig_str_landed"), f"{a}chin_score"),
            safe(pl.col(f"{col_p}_rev") / opp_ctrl_mins, f"{a}reversal_rate"),
            safe(pl.col(f"{col_p}_opp_sub_att") / pl.col(f"{col_p}_opp_td_landed"), f"{a}def_sub_exposure"),
            safe(pl.col(f"{col_p}_sig_str_landed") / pl.col(f"{col_p}_total_str_landed"), f"{a}sig_to_total_ratio"),
        ]

    # KD rate and sig_to_total: 3 windows; net_ctrl_pct: last 3 + career; style profile: career only
    output_cols = (
        ["root_fight_id", "fighter_role"]
        + ["last_fight_kd_rate", "last_3_kd_rate", "kd_rate"]
        + ["last_3_net_ctrl_pct", "net_ctrl_pct"]
        + ["last_3_sig_to_total_ratio", "sig_to_total_ratio"]
        + ["body_str_pct", "leg_str_pct"]
        + ["clinch_str_pct", "ground_str_pct"]
        + ["gnp_rate", "chin_score", "reversal_rate", "def_sub_exposure"]
    )

    return (
        pf
        .group_by(["root_fight_id", "fighter_role"])
        .agg(last_fight_aggs + last_3_aggs + career_aggs)
        .with_columns(derived("lf", "last_fight") + derived("l3", "last_3") + derived("ca", ""))
        .select(output_cols)
    )


def build_weight_class_fight_count(prior_fights: pl.DataFrame, fights: pl.DataFrame) -> pl.DataFrame:
    root_wc = fights.select(["root_fight_id", pl.col("weight_class").alias("root_weight_class")])

    at_wc = pl.col("weight_class") == pl.col("root_weight_class")
    win = pl.col("result") == "win"
    loss = pl.col("result") == "loss"
    ko = pl.col("method") == "kotko"
    sub = pl.col("method") == "sub"
    dec = pl.col("method").is_in(["d_unan", "d_maj", "d_split"])

    return (
        prior_fights
        .join(root_wc, on="root_fight_id")
        .group_by(["root_fight_id", "fighter_role"])
        .agg([
            at_wc.sum().alias("fights_at_weight"),
            (at_wc & win & ko).sum().alias("ko_wins_at_weight"),
            (at_wc & win & sub).sum().alias("sub_wins_at_weight"),
            (at_wc & win & dec).sum().alias("dec_wins_at_weight"),
            (at_wc & loss & ko).sum().alias("ko_losses_at_weight"),
            (at_wc & loss & sub).sum().alias("sub_losses_at_weight"),
            (at_wc & loss & dec).sum().alias("dec_losses_at_weight"),
        ])
    )


def build_round_features(prior_rounds: pl.DataFrame, prior_fights: pl.DataFrame) -> pl.DataFrame:
    # Bring in opponent_id and fight_date from prior_fights (rounds df has neither)
    fight_meta = prior_fights.select(["root_fight_id", "prior_fight_id", "fighter_role", "opponent_id", "fight_date"])
    rounds = prior_rounds.join(fight_meta, on=["root_fight_id", "prior_fight_id", "fighter_role"], how="left")

    own = rounds.filter(pl.col("fighter_id") != pl.col("opponent_id"))
    opp = rounds.filter(pl.col("fighter_id") == pl.col("opponent_id"))

    # Last 3 prior fights per (root_fight_id, fighter_role) by recency — computed once, reused for all windows
    top3 = (
        own.select(["root_fight_id", "fighter_role", "prior_fight_id", "fight_date"])
        .unique(["root_fight_id", "fighter_role", "prior_fight_id"])
        .with_columns(
            pl.col("fight_date").rank(method="ordinal", descending=True)
            .over(["root_fight_id", "fighter_role"])
            .alias("fight_rank")
        )
        .filter(pl.col("fight_rank") <= 3)
        .select(["root_fight_id", "fighter_role", "prior_fight_id"])
    )

    own_l3 = own.join(top3, on=["root_fight_id", "fighter_role", "prior_fight_id"], how="inner")
    opp_l3 = opp.join(top3, on=["root_fight_id", "fighter_role", "prior_fight_id"], how="inner")

    # Per-round comparison view: own vs opp in same round — used for dominance, last-round diff, post-KD
    def make_per_round(own_df, opp_df):
        max_rds = (
            own_df.group_by(["root_fight_id", "fighter_role", "prior_fight_id"])
            .agg(pl.col("round_number").max().alias("max_round"))
        )
        return (
            own_df.select(["root_fight_id", "fighter_role", "prior_fight_id", "round_number",
                           "sig_str_landed", "sig_str_attempts", "kd"])
            .rename({"sig_str_landed": "own_sig", "sig_str_attempts": "own_sig_att", "kd": "own_kd"})
            .join(
                opp_df.select(["root_fight_id", "fighter_role", "prior_fight_id", "round_number", "sig_str_landed", "kd"])
                .rename({"sig_str_landed": "opp_sig", "kd": "opp_kd"}),
                on=["root_fight_id", "fighter_role", "prior_fight_id", "round_number"],
                how="left"
            )
            .with_columns([
                pl.col("opp_sig").fill_null(0),
                pl.col("opp_kd").fill_null(0),
            ])
            .join(max_rds, on=["root_fight_id", "fighter_role", "prior_fight_id"], how="left")
            .with_columns([
                (pl.col("own_sig") > pl.col("opp_sig")).cast(pl.Float32).alias("rd_win"),
                (pl.col("own_sig") - pl.col("opp_sig")).cast(pl.Float32).alias("str_diff"),
            ])
        )

    per_round    = make_per_round(own, opp)
    per_round_l3 = make_per_round(own_l3, opp_l3)

    # Post-KD response: join each round with the previous round's opp_kd to flag "round after taking a KD"
    def make_post_kd(prd_df):
        prev = (
            prd_df.select(["root_fight_id", "fighter_role", "prior_fight_id", "round_number", "opp_kd"])
            .with_columns((pl.col("round_number") + 1).alias("round_number"))  # shift: this row becomes "next round"
            .rename({"opp_kd": "prev_opp_kd"})
        )
        return (
            prd_df.join(prev, on=["root_fight_id", "fighter_role", "prior_fight_id", "round_number"], how="left")
            .with_columns(pl.col("prev_opp_kd").fill_null(0))
            .filter(pl.col("prev_opp_kd") > 0)  # rounds immediately after being knocked down
        )

    post_kd    = make_post_kd(per_round)
    post_kd_l3 = make_post_kd(per_round_l3)

    r1    = pl.col("round_number") == 1
    early = pl.col("round_number") <= 2
    late  = pl.col("round_number") >= 3
    is_last = pl.col("round_number") == pl.col("max_round")

    def safe(expr):
        return pl.when(expr.is_nan() | expr.is_infinite()).then(pl.lit(0.0)).otherwise(expr)

    def pace_ratio(late_col, early_col, late_den, early_den):
        """(late_col/late_den) / (early_col/early_den) — neutral 1.0 if 0/0, 0.0 if n/0"""
        ratio = (pl.col(late_col).cast(pl.Float32) / pl.col(late_den).cast(pl.Float32)) \
              / (pl.col(early_col).cast(pl.Float32) / pl.col(early_den).cast(pl.Float32))
        return (
            pl.when(ratio.is_nan()).then(pl.lit(1.0))
            .when(ratio.is_infinite()).then(pl.lit(0.0))
            .otherwise(ratio)
        )

    def agg_own(df, s=""):
        p = f"_{s}" if s else ""
        return df.group_by(["root_fight_id", "fighter_role"]).agg([
            pl.col("sig_str_landed").filter(r1).sum().alias(f"r1_sig{p}"),
            pl.col("round_number").filter(r1).count().alias(f"r1_rds{p}"),
            pl.col("kd").filter(r1).sum().alias(f"r1_kd{p}"),
            pl.col("sig_str_landed").filter(early).sum().alias(f"early_sig{p}"),
            pl.col("sig_str_attempts").filter(early).sum().alias(f"early_sig_att{p}"),
            pl.col("body_landed").filter(early).sum().alias(f"early_body{p}"),
            pl.col("round_number").filter(early).count().alias(f"early_rds{p}"),
            pl.col("sig_str_landed").filter(late).sum().alias(f"late_sig{p}"),
            pl.col("sig_str_attempts").filter(late).sum().alias(f"late_sig_att{p}"),
            pl.col("body_landed").filter(late).sum().alias(f"late_body{p}"),
            pl.col("round_number").filter(late).count().alias(f"late_rds{p}"),
            pl.col("ctrl_time_s").filter(late).sum().alias(f"late_ctrl{p}"),
            pl.col("prior_fight_id").filter(late).n_unique().alias(f"fights_w_late{p}"),
            pl.col("td_landed").filter(late).sum().alias(f"late_td_l{p}"),
            pl.col("td_attempts").filter(late).sum().alias(f"late_td_a{p}"),
            pl.col("sub_att").filter(late).sum().alias(f"late_sub{p}"),
        ])

    def agg_opp(df, s=""):
        p = f"_{s}" if s else ""
        return df.group_by(["root_fight_id", "fighter_role"]).agg([
            pl.col("sig_str_landed").filter(early).sum().alias(f"opp_early_sig{p}"),
            pl.col("round_number").filter(early).count().alias(f"opp_early_rds{p}"),
            pl.col("sig_str_landed").filter(late).sum().alias(f"opp_late_sig{p}"),
            pl.col("round_number").filter(late).count().alias(f"opp_late_rds{p}"),
        ])

    def agg_prd(df, s=""):
        pfx = "last_3_" if s == "l3" else ""
        return df.group_by(["root_fight_id", "fighter_role"]).agg([
            pl.col("rd_win").mean().fill_nan(0.5).alias(f"{pfx}rd_dom_rate"),
            pl.col("str_diff").filter(is_last).mean().fill_nan(0.0).alias(f"{pfx}last_rd_str_diff"),
        ])

    def agg_post_kd(df, s=""):
        pfx = "last_3_" if s == "l3" else ""
        # own_sig in rounds after taking a KD vs career avg own_sig per round
        return df.group_by(["root_fight_id", "fighter_role"]).agg([
            pl.col("own_sig").sum().alias(f"{pfx}post_kd_sig"),
            pl.col("round_number").count().alias(f"{pfx}post_kd_rds"),
        ])

    own_agg_ca   = agg_own(own)
    own_agg_l3   = agg_own(own_l3, "l3")
    opp_agg_ca   = agg_opp(opp)
    opp_agg_l3   = agg_opp(opp_l3, "l3")
    prd_ca       = agg_prd(per_round)
    prd_l3       = agg_prd(per_round_l3, "l3")
    pkd_ca       = agg_post_kd(post_kd)
    pkd_l3       = agg_post_kd(post_kd_l3, "l3")

    # Career avg sig per round (for post-KD ratio denominator)
    career_avg = (
        own.group_by(["root_fight_id", "fighter_role"])
        .agg(pl.col("sig_str_landed").mean().alias("career_sig_per_round"))
    )

    combined = (
        own_agg_ca
        .join(own_agg_l3,  on=["root_fight_id", "fighter_role"], how="left")
        .join(opp_agg_ca,  on=["root_fight_id", "fighter_role"], how="left")
        .join(opp_agg_l3,  on=["root_fight_id", "fighter_role"], how="left")
        .join(prd_ca,      on=["root_fight_id", "fighter_role"], how="left")
        .join(prd_l3,      on=["root_fight_id", "fighter_role"], how="left")
        .join(pkd_ca,      on=["root_fight_id", "fighter_role"], how="left")
        .join(pkd_l3,      on=["root_fight_id", "fighter_role"], how="left")
        .join(career_avg,  on=["root_fight_id", "fighter_role"], how="left")
    )

    output_cols = [
        "root_fight_id", "fighter_role",
        # Career
        "r1_sig_per_fight", "r1_kd_rate",
        "slpm_pace_ratio", "sapm_pace_ratio",
        "str_acc_degradation", "body_escalation",
        "late_ctrl_per_fight", "late_td_acc", "late_sub_per_round",
        "rd_dom_rate", "last_rd_str_diff",
        "post_kd_response",
        # Last 3
        "last_3_r1_sig_per_fight",
        "last_3_slpm_pace_ratio", "last_3_sapm_pace_ratio",
        "last_3_str_acc_degradation",
        "last_3_rd_dom_rate", "last_3_last_rd_str_diff",
        "last_3_post_kd_response",
    ]

    return (
        combined
        .with_columns([
            # --- Career ---
            safe(pl.col("r1_sig").cast(pl.Float32) / pl.col("r1_rds").cast(pl.Float32)).alias("r1_sig_per_fight"),
            safe(pl.col("r1_kd").cast(pl.Float32) / pl.col("r1_sig").cast(pl.Float32)).alias("r1_kd_rate"),
            pace_ratio("late_sig", "early_sig", "late_rds", "early_rds").alias("slpm_pace_ratio"),
            pace_ratio("opp_late_sig", "opp_early_sig", "opp_late_rds", "opp_early_rds").alias("sapm_pace_ratio"),
            pace_ratio("late_sig", "early_sig", "late_sig_att", "early_sig_att").alias("str_acc_degradation"),
            pace_ratio("late_body", "early_body", "late_rds", "early_rds").alias("body_escalation"),
            safe(pl.col("late_ctrl").cast(pl.Float32) / pl.col("fights_w_late").cast(pl.Float32)).alias("late_ctrl_per_fight"),
            safe(pl.col("late_td_l").cast(pl.Float32) / pl.col("late_td_a").cast(pl.Float32)).alias("late_td_acc"),
            safe(pl.col("late_sub").cast(pl.Float32) / pl.col("late_rds").cast(pl.Float32)).alias("late_sub_per_round"),
            # post_kd_response: avg sig per post-KD round / career avg sig per round — > 1 = bounces back
            safe(
                (pl.col("post_kd_sig").cast(pl.Float32) / pl.col("post_kd_rds").cast(pl.Float32))
                / pl.col("career_sig_per_round")
            ).alias("post_kd_response"),
            # --- Last 3 ---
            safe(pl.col("r1_sig_l3").cast(pl.Float32) / pl.col("r1_rds_l3").cast(pl.Float32)).alias("last_3_r1_sig_per_fight"),
            pace_ratio("late_sig_l3", "early_sig_l3", "late_rds_l3", "early_rds_l3").alias("last_3_slpm_pace_ratio"),
            pace_ratio("opp_late_sig_l3", "opp_early_sig_l3", "opp_late_rds_l3", "opp_early_rds_l3").alias("last_3_sapm_pace_ratio"),
            pace_ratio("late_sig_l3", "early_sig_l3", "late_sig_att_l3", "early_sig_att_l3").alias("last_3_str_acc_degradation"),
            safe(
                (pl.col("last_3_post_kd_sig").cast(pl.Float32) / pl.col("last_3_post_kd_rds").cast(pl.Float32))
                / pl.col("career_sig_per_round")
            ).alias("last_3_post_kd_response"),
        ])
        .select(output_cols)
    )


def build_streak(prior_fights: pl.DataFrame) -> pl.DataFrame:
    """Current consecutive win/loss streak for each fighter.

    Sorts fights most-recent-first, then uses cumulative sums to identify
    the unbroken run at the head of the sequence:
      - win_streak  = wins before the first loss
      - loss_streak = losses before the first win
    """
    is_win = (pl.col("result") == "win").cast(pl.Int32)
    is_loss = (pl.col("result") == "loss").cast(pl.Int32)

    pf = (
        prior_fights
        .sort(["root_fight_id", "fighter_role", "fight_date"], descending=[False, False, True])
        .with_columns([
            is_win.alias("is_win"),
            is_loss.alias("is_loss"),
        ])
        .with_columns([
            pl.col("is_loss").cum_sum().over(["root_fight_id", "fighter_role"]).alias("cumsum_loss"),
            pl.col("is_win").cum_sum().over(["root_fight_id", "fighter_role"]).alias("cumsum_win"),
        ])
    )

    return (
        pf.group_by(["root_fight_id", "fighter_role"])
        .agg([
            pl.col("is_win").filter(pl.col("cumsum_loss") == 0).sum().cast(pl.Int16).alias("win_streak"),
            pl.col("is_loss").filter(pl.col("cumsum_win") == 0).sum().cast(pl.Int16).alias("loss_streak"),
        ])
    )


def split_by_role(df: pl.DataFrame, role: str, prefix: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("fighter_role") == role)
        .drop("fighter_role")
        .rename({col: f"{prefix}_{col}" for col in df.columns if col not in ("root_fight_id", "fighter_role")})
    )
