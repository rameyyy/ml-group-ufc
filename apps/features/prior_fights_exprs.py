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
        ])
        .with_columns([
            (pl.col("last_fight_end_time_s") / 300.0).alias("last_fight_rounds"),
            (pl.col("last_3_avg_end_time_s") / 300.0).alias("last_3_avg_rounds"),
            (pl.col("avg_end_time_s") / 300.0).alias("avg_rounds"),
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
            r3.sum().alias("3rd_fights"),
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
                pl.col("fight_date") >= (pl.col("root_fight_date") - pl.duration(days=730))
            ).count().alias("fights_last_2yrs"),
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
        .select(["root_fight_id", "fighter_role", "avg_fights_per_year", "fights_this_year", "fights_last_2yrs", "fights_last_3yrs"])
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
        ])
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
            (at_wc & win).sum().alias("wins_at_weight"),
            (at_wc & loss).sum().alias("losses_at_weight"),
            (at_wc & win & ko).sum().alias("ko_wins_at_weight"),
            (at_wc & win & sub).sum().alias("sub_wins_at_weight"),
            (at_wc & win & dec).sum().alias("dec_wins_at_weight"),
            (at_wc & loss & ko).sum().alias("ko_losses_at_weight"),
            (at_wc & loss & sub).sum().alias("sub_losses_at_weight"),
            (at_wc & loss & dec).sum().alias("dec_losses_at_weight"),
        ])
    )


def split_by_role(df: pl.DataFrame, role: str, prefix: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("fighter_role") == role)
        .drop("fighter_role")
        .rename({col: f"{prefix}_{col}" for col in df.columns if col not in ("root_fight_id", "fighter_role")})
    )
