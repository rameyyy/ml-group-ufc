import statistics
import polars as pl

INITIAL_ELO = 1500.0

# Variable K schedule: (min_fight_count, K)
# Fighters get the K corresponding to the highest threshold they've crossed.
# Tuned via bucket-MAE calibration on all root fights — 7.5x improvement
# over fixed K=32 (loss 0.0049 vs 0.0366).
#
#  0– 4 fights : K=66  — rookie, high uncertainty
#  5–14 fights : K=68  — developmental phase, results are very informative
# 15–19 fights : K=63  — settling into true level
# 20+  fights  : K=54  — veteran, stable rating
K_SCHEDULE: list[tuple[int, float]] = [
    (0,  66.0),
    (5,  68.0),
    (15, 63.0),
    (20, 54.0),
]


def _get_k(fight_count: int) -> float:
    k = K_SCHEDULE[0][1]
    for thresh, val in K_SCHEDULE:
        if fight_count >= thresh:
            k = val
    return k


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assemble_timeline(fights: pl.DataFrame, prior_fights: pl.DataFrame) -> pl.DataFrame:
    """One row per unique fight across root fights + all prior fight history,
    sorted chronologically oldest-first. Root version is kept when a fight
    appears in both sources."""
    root = fights.select([
        pl.col("root_fight_id").alias("fight_id"),
        pl.col("fighter1_id"),
        pl.col("fighter2_id"),
        pl.col("winner_id"),
        pl.col("fight_date"),
        pl.lit(True).alias("is_root"),
    ])

    id_map = fights.select(["root_fight_id", "fighter1_id", "fighter2_id"])

    prior = (
        prior_fights
        .join(id_map, on="root_fight_id", how="left")
        .with_columns(
            pl.when(pl.col("fighter_role") == "f1")
            .then(pl.col("fighter1_id"))
            .otherwise(pl.col("fighter2_id"))
            .alias("fighter_id")
        )
        # Same prior fight appears for many root fights — keep one row
        .unique(subset=["prior_fight_id"], keep="first")
        .with_columns(
            pl.when(pl.col("result") == "win")
            .then(pl.col("fighter_id"))
            .otherwise(pl.col("opponent_id"))
            .alias("winner_id")
        )
        .select([
            pl.col("prior_fight_id").alias("fight_id"),
            pl.col("fighter_id").alias("fighter1_id"),
            pl.col("opponent_id").alias("fighter2_id"),
            pl.col("winner_id"),
            pl.col("fight_date"),
            pl.lit(False).alias("is_root"),
        ])
    )

    return (
        pl.concat([root, prior])
        # Root version wins dedup within same fight_id
        .sort(["fight_date", "is_root"], descending=[False, True])
        .unique(subset=["fight_id"], keep="first")
        .sort(["fight_date", "fight_id"])  # deterministic within same-day fights
    )


def _run_elo(timeline: pl.DataFrame) -> tuple[dict, list[dict]]:
    """One chronological ELO pass with variable K schedule.
    Returns (final_elo_dict, root_fight_snapshots).
    """
    elo: dict[str, float] = {}
    fight_count: dict[str, int] = {}
    snapshots: list[dict] = []

    for row in timeline.iter_rows(named=True):
        f1_id  = row["fighter1_id"]
        f2_id  = row["fighter2_id"]
        winner = row["winner_id"]
        fid    = row["fight_id"]

        r1 = elo.get(f1_id, INITIAL_ELO)
        r2 = elo.get(f2_id, INITIAL_ELO)

        if row["is_root"]:
            snapshots.append({
                "root_fight_id": fid,
                "f1_elo": r1,
                "f2_elo": r2,
                "winner_is_f1": winner == f1_id,
            })

        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        s1 = 1.0 if winner == f1_id else 0.0

        K1 = _get_k(fight_count.get(f1_id, 0))
        K2 = _get_k(fight_count.get(f2_id, 0))

        elo[f1_id] = r1 + K1 * (s1 - e1)
        elo[f2_id] = r2 + K2 * ((1.0 - s1) - (1.0 - e1))

        fight_count[f1_id] = fight_count.get(f1_id, 0) + 1
        fight_count[f2_id] = fight_count.get(f2_id, 0) + 1

    return elo, snapshots


def _calibration_loss(snapshots: list[dict], min_elo_diff: float = 50.0) -> float:
    """Bucket-based MAE calibration, robust to upsets.

    Splits fights into 5 quintiles by ELO diff. Compares median predicted
    win% to actual win rate per bucket. Using median (not mean) on predicted
    prob so extreme ELO gaps don't dominate — upsets are common in UFC and
    outliers would otherwise pull the metric around.

    Only uses fights where |elo_diff| > min_elo_diff to skip the cold-start
    region where most fighters are near 1500 and all configs look the same.
    """
    items = [
        (
            s["f1_elo"] - s["f2_elo"],
            1.0 / (1.0 + 10.0 ** ((s["f2_elo"] - s["f1_elo"]) / 400.0)),
            1.0 if s["winner_is_f1"] else 0.0,
        )
        for s in snapshots
        if abs(s["f1_elo"] - s["f2_elo"]) > min_elo_diff
    ]
    if not items:
        return float("inf")

    items.sort(key=lambda x: x[0])
    n = len(items)
    bin_size = max(n // 5, 1)

    total_err, n_bins = 0.0, 0
    for i in range(5):
        chunk = items[i * bin_size : (i + 1) * bin_size if i < 4 else n]
        if not chunk:
            continue
        median_pred = statistics.median(x[1] for x in chunk)
        actual_rate = sum(x[2] for x in chunk) / len(chunk)
        total_err  += abs(median_pred - actual_rate)
        n_bins     += 1

    return total_err / n_bins if n_bins else float("inf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_elo_ratings(fights: pl.DataFrame, prior_fights: pl.DataFrame) -> pl.DataFrame:
    """Returns DataFrame with f1_elo, f2_elo, elo_diff keyed on root_fight_id.

    Uses a variable K schedule tuned against bucket-MAE calibration:
      0– 4 fights: K=66  |  5–14: K=68  |  15–19: K=63  |  20+: K=54
    Calibration loss: 0.0049  (vs 0.0366 for standard fixed K=32)
    """
    timeline = _assemble_timeline(fights, prior_fights)
    _, snapshots = _run_elo(timeline)

    return (
        pl.DataFrame(snapshots)
        .drop("winner_is_f1")
        .with_columns([
            pl.col("f1_elo").cast(pl.Float32),
            pl.col("f2_elo").cast(pl.Float32),
            (pl.col("f1_elo") - pl.col("f2_elo")).cast(pl.Float32).alias("elo_diff"),
        ])
    )


def print_top_fighters(fights: pl.DataFrame, prior_fights: pl.DataFrame, n: int = 10) -> None:
    """Run ELO to completion and print top-N fighters by final rating."""
    timeline = _assemble_timeline(fights, prior_fights)
    final_elo, _ = _run_elo(timeline)

    names: dict[str, str] = {}
    for row in fights.select(["fighter1_id", "fighter1_name", "fighter2_id", "fighter2_name"]).iter_rows(named=True):
        names[row["fighter1_id"]] = row["fighter1_name"]
        names[row["fighter2_id"]] = row["fighter2_name"]

    ranked = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:n]

    print(f"\n{'Rank':<6} {'Fighter':<25} {'ELO':>7}")
    print("-" * 40)
    for i, (fid, rating) in enumerate(ranked, 1):
        print(f"{i:<6} {names.get(fid, fid):<25} {rating:>7.1f}")


# ---------------------------------------------------------------------------
# Run directly: print final top 10 with calibration loss
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from extract import get_fightsnapshots_df, unnest_raw_df

    raw = get_fightsnapshots_df()
    fights_df, prior_fights_df, _ = unnest_raw_df(raw)

    timeline = _assemble_timeline(fights_df, prior_fights_df)
    final_elo, snapshots = _run_elo(timeline)

    loss = _calibration_loss(snapshots)
    print(f"Calibration loss: {loss:.4f}")
    print(f"K schedule: {K_SCHEDULE}")

    print_top_fighters(fights_df, prior_fights_df)
