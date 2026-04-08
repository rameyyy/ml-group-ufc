import statistics
import polars as pl

INITIAL_ELO  = 1500.0
K_DOMAIN     = 40.0    # fixed K for striking/grappling/finishing domain ELOs
GRAP_TD_WEIGHT = 30    # seconds of ctrl-time equivalent per takedown landed

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

# ---------------------------------------------------------------------------
# Weight-class transfer system
# ---------------------------------------------------------------------------

# Canonical weight limits per division (lbs).
# Used to compute weight-class proximity when a fighter changes division.
WEIGHT_CLASS_LBS: dict[str, float] = {
    "strawweight":       115.0,
    "flyweight":         125.0,
    "bantamweight":      135.0,
    "featherweight":     145.0,
    "lightweight":       155.0,
    "welterweight":      170.0,
    "middleweight":      185.0,
    "light heavyweight": 205.0,
    "heavyweight":       265.0,
    "catch weight":      160.0,   # approximate midpoint
}
_MAX_WEIGHT_DIFF = max(WEIGHT_CLASS_LBS.values()) - min(WEIGHT_CLASS_LBS.values())  # 150

# Per-division transfer factors: how much of the global ELO surplus (above 1500)
# carries into a new division.  Scaled further by weight-class proximity so
# adjacent moves transfer more than long jumps.
#   entry_elo = 1500 + (global_elo - 1500) * transfer_factor(from_div, to_div)
#
# Values calibrated via per-division backtest (accuracy of div ELO predicting
# fight outcomes on held-out root fights, varying base 0.0→1.0):
#
#   flyweight         1.0  — thin division, carry-over helps
#   bantamweight      0.2  — deep + gender mixing; low transfer reduces noise
#   featherweight     1.0  — carry-over wins clearly
#   lightweight       1.0  — carry-over wins clearly
#   welterweight      0.0  — deepest division, fully self-contained
#   middleweight      0.2  — deep, self-contained; low transfer wins
#   light heavyweight 1.0  — many elite crossovers (Jones, Pereira); full carry-over
#   heavyweight       0.4  — small division, partial carry-over best
TRANSFER_BASE = 0.4   # fallback for unknown divisions

TRANSFER_BY_DIV: dict[str, float] = {
    "flyweight":         1.0,
    "bantamweight":      0.2,
    "featherweight":     1.0,
    "lightweight":       1.0,
    "welterweight":      0.0,
    "middleweight":      0.2,
    "light heavyweight": 1.0,
    "heavyweight":       0.4,
    "catch weight":      0.4,
}

# ---------------------------------------------------------------------------
# ELO decay (display / rankings only — NOT used for ML features)
# ---------------------------------------------------------------------------
DECAY_GRACE_MONTHS  = 6    # no decay within first 6 inactive months
DECAY_POINTS_MONTH  = 2.5  # points lost per inactive month after grace period


def _get_k(fight_count: int) -> float:
    k = K_SCHEDULE[0][1]
    for thresh, val in K_SCHEDULE:
        if fight_count >= thresh:
            k = val
    return k


def _transfer_factor(from_div: str, to_div: str) -> float:
    """Fraction of global ELO surplus that carries into a new division.

    Base is looked up per destination division from TRANSFER_BY_DIV (calibrated
    via backtest).  Then scaled by weight-class proximity so adjacent moves
    (MW→LHW) carry more than long jumps (FW→LHW).

        factor = base(to_div) * (1 - |from_lbs - to_lbs| / max_weight_diff)

    Falls back to TRANSFER_BASE when either division is unknown.
    """
    base     = TRANSFER_BY_DIV.get(to_div, TRANSFER_BASE)
    from_lbs = WEIGHT_CLASS_LBS.get(from_div)
    to_lbs   = WEIGHT_CLASS_LBS.get(to_div)
    if from_lbs is None or to_lbs is None:
        return base
    diff = abs(from_lbs - to_lbs)
    return base * (1.0 - diff / _MAX_WEIGHT_DIFF)


# ---------------------------------------------------------------------------
# Fight-stats lookup (used by domain ELO systems)
# ---------------------------------------------------------------------------

def _parse_end_time_s(s) -> float:
    """Parse 'MM:SS' total-fight-time string to seconds. Returns 0 on failure."""
    try:
        parts = str(s).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


def _build_fight_stats(
    prior_fights_df: pl.DataFrame,
    fights_df: pl.DataFrame,
) -> dict:
    """Return {fight_id: stats_dict} for every fight that has per-round stats.

    Stats are stored from the perspective of fighter1_id as used in the
    assembled timeline (where fighter1_id = the fighter whose prior-fight row
    this is, and fighter2_id = opponent_id).  For root fights we only have the
    weight_class — per-fight striking stats aren't in fights_df directly.
    """
    lookup: dict = {}

    # --- prior fights: full per-fight stats available ---
    for row in (
        prior_fights_df
        .unique(subset=["prior_fight_id"], keep="first")
        .iter_rows(named=True)
    ):
        fid = row["prior_fight_id"]
        if fid in lookup:
            continue
        end_s = _parse_end_time_s(row.get("end_time"))
        lookup[fid] = {
            # striking
            "f1_sig":    row["sig_str_landed"]     or 0,
            "f2_sig":    row["opp_sig_str_landed"]  or 0,
            # grappling
            "f1_td":     row["td_landed"]           or 0,
            "f2_td":     row["opp_td_landed"]       or 0,
            "f1_ctrl":   row["ctrl_time_s"]         or 0,
            "f2_ctrl":   row["opp_ctrl_time_s"]     or 0,
            # finishing
            "method":    (row["method"] or "").lower(),
            "end_time_s": end_s,
            "weight_class": (row["weight_class"] or "").lower(),
        }

    # --- root fights: only weight_class available (no per-fight strike stats) ---
    for row in fights_df.select(["root_fight_id", "weight_class"]).iter_rows(named=True):
        fid = row["root_fight_id"]
        if fid not in lookup:
            # Insert a stub so division ELO can still be updated
            lookup[fid] = {
                "f1_sig": None, "f2_sig": None,
                "f1_td":  None, "f2_td":  None,
                "f1_ctrl": None, "f2_ctrl": None,
                "method": None, "end_time_s": None,
                "weight_class": (row["weight_class"] or "").lower(),
            }

    return lookup


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


PEAK_LAYOFF_DAYS   = 365  # inactive longer than this resets the peak window
PEAK_LOSS_STREAK   = 2    # this many consecutive losses resets the peak window


def _run_elo(timeline: pl.DataFrame) -> tuple[dict, list[dict]]:
    """One chronological ELO pass with variable K schedule.
    Returns (final_elo_dict, root_fight_snapshots).

    Peak ELO logic
    --------------
    Each fighter has a "peak window" — a continuous stretch of activity with
    no long layoff and no significant loss streak. Within a window the peak
    ELO is the running max. The window resets when:
      - gap since last fight > PEAK_LAYOFF_DAYS (365 days), OR
      - PEAK_LOSS_STREAK (2) consecutive losses just occurred.
    On reset the peak restarts from the fighter's current ELO; the old window
    peak is discarded. This means a fighter who peaked at 1700, dropped two
    fights, then won back to 1651 has peak_elo=1651 (new window) rather than
    carrying the stale 1700 forward.
    """
    elo: dict[str, float] = {}
    fight_count: dict[str, int] = {}
    consec_losses: dict[str, int] = {}
    last_fight_date: dict = {}   # fighter_id -> datetime.date
    window_peak: dict[str, float] = {}
    snapshots: list[dict] = []

    for row in timeline.iter_rows(named=True):
        f1_id  = row["fighter1_id"]
        f2_id  = row["fighter2_id"]
        winner = row["winner_id"]
        fid    = row["fight_id"]
        fd     = row["fight_date"]   # datetime.date (pl.Date col)

        r1 = elo.get(f1_id, INITIAL_ELO)
        r2 = elo.get(f2_id, INITIAL_ELO)

        # --- peak window update (pre-fight, no leakage) ---
        for fid_p, curr in ((f1_id, r1), (f2_id, r2)):
            reset = (
                consec_losses.get(fid_p, 0) >= PEAK_LOSS_STREAK
                or (
                    fid_p in last_fight_date
                    and (fd - last_fight_date[fid_p]).days > PEAK_LAYOFF_DAYS
                )
            )
            if reset:
                window_peak[fid_p] = curr
                consec_losses[fid_p] = 0          # fresh window, clear streak
            elif fid_p not in window_peak:
                window_peak[fid_p] = curr
            else:
                window_peak[fid_p] = max(window_peak[fid_p], curr)

        peak1 = window_peak[f1_id]
        peak2 = window_peak[f2_id]

        if row["is_root"]:
            snapshots.append({
                "root_fight_id": fid,
                "f1_elo": r1,
                "f2_elo": r2,
                "f1_peak_elo": peak1,
                "f2_peak_elo": peak2,
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

        # update loss streak and last fight date post-fight
        if winner == f1_id:
            consec_losses[f1_id] = 0
            consec_losses[f2_id] = consec_losses.get(f2_id, 0) + 1
        else:
            consec_losses[f1_id] = consec_losses.get(f1_id, 0) + 1
            consec_losses[f2_id] = 0

        last_fight_date[f1_id] = fd
        last_fight_date[f2_id] = fd

    return elo, snapshots


def _elo_update(ra: float, rb: float, s_a: float, k: float) -> tuple[float, float]:
    """Return (new_ra, new_rb) after one ELO exchange."""
    e_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    new_ra = ra + k * (s_a - e_a)
    new_rb = rb + k * ((1.0 - s_a) - (1.0 - e_a))
    return new_ra, new_rb


def _run_all_elos(
    timeline: pl.DataFrame,
    fight_stats: dict,
) -> tuple[dict, list[dict]]:
    """Full ELO pass tracking overall + division + striking + grappling + finishing ELOs.

    Returns (final_elo_dict, root_fight_snapshots, pre_fight_elos, elo_after_each_fight).

    pre_fight_elos         {fight_id: {fighter_id: elo_before_fight}} — used by
                           build_schedule_strength_features to look up opponent ELO
                           at the time of each prior fight.
    elo_after_each_fight   {fighter_id: [elo_after_fight_0, elo_after_fight_1, ...]}
                           chronological post-fight ELO history used for velocity
                           features (elo_delta_last_3 / elo_delta_last_5).

    Domain ELO systems
    ------------------
    All domain ELOs start at INITIAL_ELO and use a fixed K=K_DOMAIN.

    Division ELO
        Separate ELO per weight class.  Initialised at the fighter's current
        overall ELO the first time they fight at a new division (carry-over
        prior).  Only the fight's weight class ELO is updated.

    Striking offense / defense  (str_off / str_def)
        Binary outcome: did fighter1 land more sig strikes per minute?
        str_off[A] vs str_def[B] for A's attack; str_off[B] vs str_def[A]
        for B's attack.  Both updated simultaneously per fight.
        Falls back to win/loss when strike stats unavailable.

    Grappling offense / defense  (grap_off / grap_def)
        Binary outcome: did fighter1 win grappling exchange?
        Composite score = td_landed * GRAP_TD_WEIGHT + ctrl_time_s.
        Same paired update pattern as striking.
        Falls back to win/loss when grappling stats unavailable.

    Finishing / Durability  (finish_off / durability)
        Binary outcome: did fighter1 finish the fight (KO/TKO/SUB = 1)?
        finish_off[A] vs durability[B]: did A finish B?
        finish_off[B] vs durability[A]: did B finish A?
        Falls back to no-op when method unavailable.
    """
    elo:          dict[str, float] = {}
    fight_count:  dict[str, int]   = {}   # global fight count (drives global K)

    # Domain ELOs
    div_elo:       dict[str, dict]  = {}   # fighter_id -> {weight_class: elo}
    div_fight_count: dict[str, dict]= {}   # fighter_id -> {weight_class: count}
    last_div:      dict[str, str]   = {}   # fighter_id -> most recent weight class
    str_off:     dict[str, float]  = {}
    str_def:     dict[str, float]  = {}
    grap_off:    dict[str, float]  = {}
    grap_def:    dict[str, float]  = {}
    finish_off:  dict[str, float]  = {}
    durability:  dict[str, float]  = {}

    # Peak-window tracking (same as _run_elo)
    consec_losses:   dict[str, int]  = {}
    last_fight_date: dict            = {}
    window_peak:     dict[str, float]= {}

    snapshots: list[dict] = []
    pre_fight_elos:       dict[str, dict[str, float]] = {}  # fight_id -> {fid: elo_before}
    elo_after_each_fight: dict[str, list[float]]      = {}  # fighter_id -> [elo_after_0, ...]

    for row in timeline.iter_rows(named=True):
        f1_id  = row["fighter1_id"]
        f2_id  = row["fighter2_id"]
        winner = row["winner_id"]
        fid    = row["fight_id"]
        fd     = row["fight_date"]

        r1 = elo.get(f1_id, INITIAL_ELO)
        r2 = elo.get(f2_id, INITIAL_ELO)
        fc1 = fight_count.get(f1_id, 0)
        fc2 = fight_count.get(f2_id, 0)

        # --- peak window update (no leakage) ---
        for fp, curr in ((f1_id, r1), (f2_id, r2)):
            reset = (
                consec_losses.get(fp, 0) >= PEAK_LOSS_STREAK
                or (fp in last_fight_date
                    and (fd - last_fight_date[fp]).days > PEAK_LAYOFF_DAYS)
            )
            if reset:
                window_peak[fp] = curr
                consec_losses[fp] = 0
            elif fp not in window_peak:
                window_peak[fp] = curr
            else:
                window_peak[fp] = max(window_peak[fp], curr)

        peak1 = window_peak[f1_id]
        peak2 = window_peak[f2_id]

        # --- division ELO init with dynamic transfer factor ---
        # When a fighter enters a new division their seed ELO is:
        #   1500 + (global_elo - 1500) * transfer_factor(prev_div, new_div)
        # This discounts the global surplus based on how far they're moving.
        # Adjacent moves (MW→LHW) transfer more; long jumps (FW→LHW) transfer less.
        # K for the division resets to the variable schedule using division fight
        # count — so a dominant MW entering LHW gets high K (fast adjustment)
        # even though their global K is low (stable global rating).
        stats = fight_stats.get(fid, {})
        wc    = stats.get("weight_class") or ""
        for fp, curr in ((f1_id, r1), (f2_id, r2)):
            if fp not in div_elo:
                div_elo[fp] = {}
            if fp not in div_fight_count:
                div_fight_count[fp] = {}
            if wc and wc not in div_elo[fp]:
                prev = last_div.get(fp)
                if prev:
                    factor = _transfer_factor(prev, wc)
                else:
                    # First ever division: use destination base directly
                    # (no proximity scaling — no prior division to measure from)
                    factor = TRANSFER_BY_DIV.get(wc, TRANSFER_BASE)
                surplus = elo.get(fp, INITIAL_ELO) - INITIAL_ELO
                div_elo[fp][wc] = INITIAL_ELO + surplus * factor

        div1 = div_elo[f1_id].get(wc, INITIAL_ELO) if wc else INITIAL_ELO
        div2 = div_elo[f2_id].get(wc, INITIAL_ELO) if wc else INITIAL_ELO

        # --- domain pre-fight ELOs ---
        so1 = str_off.get(f1_id,   INITIAL_ELO)
        so2 = str_off.get(f2_id,   INITIAL_ELO)
        sd1 = str_def.get(f1_id,   INITIAL_ELO)
        sd2 = str_def.get(f2_id,   INITIAL_ELO)
        go1 = grap_off.get(f1_id,  INITIAL_ELO)
        go2 = grap_off.get(f2_id,  INITIAL_ELO)
        gd1 = grap_def.get(f1_id,  INITIAL_ELO)
        gd2 = grap_def.get(f2_id,  INITIAL_ELO)
        fo1 = finish_off.get(f1_id, INITIAL_ELO)
        fo2 = finish_off.get(f2_id, INITIAL_ELO)
        du1 = durability.get(f1_id, INITIAL_ELO)
        du2 = durability.get(f2_id, INITIAL_ELO)

        # Record pre-fight ELOs for all fights (used by schedule strength)
        pre_fight_elos[fid] = {f1_id: r1, f2_id: r2}

        if row["is_root"]:
            snapshots.append({
                "root_fight_id":   fid,
                # temporary fields popped in build_elo_ratings for velocity calc
                "_f1_id": f1_id, "_f2_id": f2_id,
                "_f1_idx": len(elo_after_each_fight.get(f1_id, [])),
                "_f2_idx": len(elo_after_each_fight.get(f2_id, [])),
                "f1_elo":          r1,    "f2_elo":          r2,
                "f1_peak_elo":     peak1, "f2_peak_elo":     peak2,
                "f1_elo_at_weight": div1, "f2_elo_at_weight": div2,
                "f1_str_off_elo":  so1,   "f2_str_off_elo":  so2,
                "f1_str_def_elo":  sd1,   "f2_str_def_elo":  sd2,
                "f1_grap_off_elo": go1,   "f2_grap_off_elo": go2,
                "f1_grap_def_elo": gd1,   "f2_grap_def_elo": gd2,
                "f1_finish_elo":   fo1,   "f2_finish_elo":   fo2,
                "f1_durability_elo": du1, "f2_durability_elo": du2,
                "winner_is_f1":    winner == f1_id,
            })

        # ---------------------------------------------------------------
        # Overall ELO update
        # ---------------------------------------------------------------
        s1  = 1.0 if winner == f1_id else 0.0
        K1  = _get_k(fc1)
        K2  = _get_k(fc2)
        new_r1, new_r2 = _elo_update(r1, r2, s1, K1)
        # Use symmetric K for both fighters (K2 for f2)
        new_r2 = r2 + K2 * ((1.0 - s1) - (1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))))
        elo[f1_id] = new_r1
        elo[f2_id] = new_r2

        # ---------------------------------------------------------------
        # Division ELO update
        # K uses division fight count (not global) so new entrants adjust
        # quickly regardless of how many fights they have globally.
        # ---------------------------------------------------------------
        if wc:
            div_fc1 = div_fight_count[f1_id].get(wc, 0)
            div_fc2 = div_fight_count[f2_id].get(wc, 0)
            Kd1 = _get_k(div_fc1)
            Kd2 = _get_k(div_fc2)
            e_d1 = 1.0 / (1.0 + 10.0 ** ((div2 - div1) / 400.0))
            div_elo[f1_id][wc] = div1 + Kd1 * (s1 - e_d1)
            div_elo[f2_id][wc] = div2 + Kd2 * ((1.0 - s1) - (1.0 - e_d1))

        # ---------------------------------------------------------------
        # Striking ELO update
        # Outcome: did f1 land more sig strikes per minute?
        # ---------------------------------------------------------------
        f1_sig   = stats.get("f1_sig")
        f2_sig   = stats.get("f2_sig")
        end_time = stats.get("end_time_s") or 0

        if f1_sig is not None and f2_sig is not None and end_time > 0:
            f1_slpm = f1_sig / (end_time / 60.0)
            f2_slpm = f2_sig / (end_time / 60.0)
            s_str = 1.0 if f1_slpm > f2_slpm else 0.0
        else:
            s_str = s1   # fallback: use overall win/loss

        # f1 offense vs f2 defense
        str_off[f1_id], str_def[f2_id] = _elo_update(so1, sd2, s_str,       K_DOMAIN)
        # f2 offense vs f1 defense
        str_off[f2_id], str_def[f1_id] = _elo_update(so2, sd1, 1.0 - s_str, K_DOMAIN)

        # ---------------------------------------------------------------
        # Grappling ELO update
        # Outcome: who had the higher grappling composite score?
        # ---------------------------------------------------------------
        f1_td   = stats.get("f1_td")
        f2_td   = stats.get("f2_td")
        f1_ctrl = stats.get("f1_ctrl")
        f2_ctrl = stats.get("f2_ctrl")

        if f1_td is not None and f2_td is not None:
            f1_grap = f1_td * GRAP_TD_WEIGHT + (f1_ctrl or 0)
            f2_grap = f2_td * GRAP_TD_WEIGHT + (f2_ctrl or 0)
            s_grap = 1.0 if f1_grap > f2_grap else (0.5 if f1_grap == f2_grap else 0.0)
        else:
            s_grap = s1   # fallback

        grap_off[f1_id], grap_def[f2_id] = _elo_update(go1, gd2, s_grap,       K_DOMAIN)
        grap_off[f2_id], grap_def[f1_id] = _elo_update(go2, gd1, 1.0 - s_grap, K_DOMAIN)

        # ---------------------------------------------------------------
        # Finishing / Durability ELO update
        # Outcome: did f1 finish the fight?
        # ---------------------------------------------------------------
        method = stats.get("method") or ""
        if method:
            f1_finished_b = winner == f1_id and ("ko" in method or "tko" in method or "sub" in method)
            f2_finished_b = winner == f2_id and ("ko" in method or "tko" in method or "sub" in method)
            s_fin_f1 = 1.0 if f1_finished_b else 0.0
            s_fin_f2 = 1.0 if f2_finished_b else 0.0
            finish_off[f1_id], durability[f2_id] = _elo_update(fo1, du2, s_fin_f1, K_DOMAIN)
            finish_off[f2_id], durability[f1_id] = _elo_update(fo2, du1, s_fin_f2, K_DOMAIN)

        # ---------------------------------------------------------------
        # Post-fight bookkeeping
        # ---------------------------------------------------------------
        fight_count[f1_id] = fc1 + 1
        fight_count[f2_id] = fc2 + 1

        if wc:
            div_fight_count[f1_id][wc] = div_fight_count[f1_id].get(wc, 0) + 1
            div_fight_count[f2_id][wc] = div_fight_count[f2_id].get(wc, 0) + 1
            last_div[f1_id] = wc
            last_div[f2_id] = wc

        if winner == f1_id:
            consec_losses[f1_id] = 0
            consec_losses[f2_id] = consec_losses.get(f2_id, 0) + 1
        else:
            consec_losses[f1_id] = consec_losses.get(f1_id, 0) + 1
            consec_losses[f2_id] = 0

        last_fight_date[f1_id] = fd
        last_fight_date[f2_id] = fd

        # Record post-fight ELOs for velocity features
        elo_after_each_fight.setdefault(f1_id, []).append(elo[f1_id])
        elo_after_each_fight.setdefault(f2_id, []).append(elo[f2_id])

    return elo, snapshots, pre_fight_elos, elo_after_each_fight


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
# Display decay (rankings only — do NOT use for ML features)
# ---------------------------------------------------------------------------

def apply_display_decay(
    div_elo_map: dict[str, dict[str, float]],
    last_fight_date_by_div: dict[str, dict[str, object]],
    as_of_date,
    grace_months: float = DECAY_GRACE_MONTHS,
    points_per_month: float = DECAY_POINTS_MONTH,
) -> dict[str, dict[str, float]]:
    """Return a decayed copy of div_elo_map for display purposes only.

    Each fighter's division ELO decays linearly after grace_months of
    inactivity at that weight class.  The floor is the fighter's entry ELO
    for that division (stored at first fight) so ratings never decay below
    where they started.

    Args:
        div_elo_map:            {fighter_id: {wc: current_elo}}
        last_fight_date_by_div: {fighter_id: {wc: last_fight_date}}
        as_of_date:             date to compute inactivity against
        grace_months:           inactive months before decay starts (default 6)
        points_per_month:       ELO points lost per month after grace (default 2.5)
    """
    import datetime
    decayed: dict[str, dict[str, float]] = {}
    for fid, divs in div_elo_map.items():
        decayed[fid] = {}
        for wc, current_elo in divs.items():
            last = (last_fight_date_by_div.get(fid) or {}).get(wc)
            if last is None:
                decayed[fid][wc] = current_elo
                continue
            months_inactive = (as_of_date - last).days / 30.44
            decay = max(0.0, (months_inactive - grace_months) * points_per_month)
            decayed[fid][wc] = current_elo - decay
    return decayed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_elo_ratings(fights: pl.DataFrame, prior_fights: pl.DataFrame) -> pl.DataFrame:
    """Returns a DataFrame of all ELO features keyed on root_fight_id.

    Overall ELO (variable K, calibrated):
      f1_elo, f2_elo, elo_diff
      f1_peak_elo, f2_peak_elo, f1_elo_vs_peak, f2_elo_vs_peak

    Division ELO (separate ELO per weight class):
      f1_elo_at_weight, f2_elo_at_weight, elo_diff_at_weight

    Striking offense / defense ELO:
      f1_str_off_elo, f2_str_off_elo, f1_str_def_elo, f2_str_def_elo
      str_matchup_diff  = (f1_off - f2_def) - (f2_off - f1_def)

    Grappling offense / defense ELO:
      f1_grap_off_elo, f2_grap_off_elo, f1_grap_def_elo, f2_grap_def_elo
      grap_matchup_diff = (f1_off - f2_def) - (f2_off - f1_def)

    Finishing / Durability ELO:
      f1_finish_elo, f2_finish_elo, f1_durability_elo, f2_durability_elo

    ELO Velocity:
      f1_elo_delta_last_3, f2_elo_delta_last_3  — ELO change over last 3 fights
      f1_elo_delta_last_5, f2_elo_delta_last_5  — ELO change over last 5 fights
      elo_delta_diff_last_3                      — f1 delta minus f2 delta
    """
    timeline    = _assemble_timeline(fights, prior_fights)
    fight_stats = _build_fight_stats(prior_fights, fights)
    _, snapshots, pre_fight_elos, elo_after_each_fight = _run_all_elos(timeline, fight_stats)

    # Compute velocity before converting snapshots to a DataFrame.
    # _f1_id/_f2_id/_f1_idx/_f2_idx are temp fields added in _run_all_elos.
    for s in snapshots:
        f1_id = s.pop("_f1_id")
        f2_id = s.pop("_f2_id")
        idx1  = s.pop("_f1_idx")
        idx2  = s.pop("_f2_idx")
        r1, r2 = s["f1_elo"], s["f2_elo"]
        h1 = elo_after_each_fight.get(f1_id, [])
        h2 = elo_after_each_fight.get(f2_id, [])
        s["f1_elo_delta_last_3"] = float(r1 - (h1[idx1 - 3] if idx1 >= 3 else r1))
        s["f1_elo_delta_last_5"] = float(r1 - (h1[idx1 - 5] if idx1 >= 5 else r1))
        s["f2_elo_delta_last_3"] = float(r2 - (h2[idx2 - 3] if idx2 >= 3 else r2))
        s["f2_elo_delta_last_5"] = float(r2 - (h2[idx2 - 5] if idx2 >= 5 else r2))

    f32 = pl.Float32
    elo_df = (
        pl.DataFrame(snapshots)
        .drop("winner_is_f1")
        .with_columns([
            # --- overall ---
            pl.col("f1_elo").cast(f32),
            pl.col("f2_elo").cast(f32),
            pl.col("f1_peak_elo").cast(f32),
            pl.col("f2_peak_elo").cast(f32),
            (pl.col("f1_elo") - pl.col("f2_elo")).cast(f32).alias("elo_diff"),
            (pl.col("f1_elo") / pl.col("f1_peak_elo")).cast(f32).alias("f1_elo_vs_peak"),
            (pl.col("f2_elo") / pl.col("f2_peak_elo")).cast(f32).alias("f2_elo_vs_peak"),
            # --- division ---
            pl.col("f1_elo_at_weight").cast(f32),
            pl.col("f2_elo_at_weight").cast(f32),
            (pl.col("f1_elo_at_weight") - pl.col("f2_elo_at_weight")).cast(f32).alias("elo_diff_at_weight"),
            # --- striking ---
            pl.col("f1_str_off_elo").cast(f32),
            pl.col("f2_str_off_elo").cast(f32),
            pl.col("f1_str_def_elo").cast(f32),
            pl.col("f2_str_def_elo").cast(f32),
            (
                (pl.col("f1_str_off_elo") - pl.col("f2_str_def_elo"))
                - (pl.col("f2_str_off_elo") - pl.col("f1_str_def_elo"))
            ).cast(f32).alias("str_matchup_diff"),
            # --- grappling ---
            pl.col("f1_grap_off_elo").cast(f32),
            pl.col("f2_grap_off_elo").cast(f32),
            pl.col("f1_grap_def_elo").cast(f32),
            pl.col("f2_grap_def_elo").cast(f32),
            (
                (pl.col("f1_grap_off_elo") - pl.col("f2_grap_def_elo"))
                - (pl.col("f2_grap_off_elo") - pl.col("f1_grap_def_elo"))
            ).cast(f32).alias("grap_matchup_diff"),
            # --- finishing ---
            pl.col("f1_finish_elo").cast(f32),
            pl.col("f2_finish_elo").cast(f32),
            pl.col("f1_durability_elo").cast(f32),
            pl.col("f2_durability_elo").cast(f32),
            # --- velocity ---
            pl.col("f1_elo_delta_last_3").cast(f32),
            pl.col("f2_elo_delta_last_3").cast(f32),
            pl.col("f1_elo_delta_last_5").cast(f32),
            pl.col("f2_elo_delta_last_5").cast(f32),
            (pl.col("f1_elo_delta_last_3") - pl.col("f2_elo_delta_last_3")).cast(f32).alias("elo_delta_diff_last_3"),
        ])
    )

    # --- Strength-of-Schedule features ---
    # Look up pre-fight ELO of each opponent at the time they fought each fighter.
    # pre_fight_elos[fight_id][fighter_id] = ELO before that fight.
    sos_rows = []
    for row in prior_fights.select(
        ["root_fight_id", "fighter_role", "prior_fight_id", "opponent_id"]
    ).iter_rows(named=True):
        pfid    = row["prior_fight_id"]
        opp     = row["opponent_id"]
        opp_elo = pre_fight_elos.get(pfid, {}).get(opp)
        if opp_elo is not None:
            sos_rows.append({
                "root_fight_id": row["root_fight_id"],
                "fighter_role":  row["fighter_role"],
                "opp_elo":       float(opp_elo),
            })

    if sos_rows:
        sos_agg = (
            pl.DataFrame(sos_rows)
            .group_by(["root_fight_id", "fighter_role"])
            .agg([
                pl.col("opp_elo").mean().cast(f32).alias("avg_opp_elo"),
                (pl.col("opp_elo") > ELITE_OPP_ELO_THRESHOLD)
                .mean().cast(f32).alias("pct_elite_opps"),
            ])
        )
        f1_sos = (
            sos_agg.filter(pl.col("fighter_role") == "f1")
            .drop("fighter_role")
            .rename({"avg_opp_elo": "f1_avg_opp_elo", "pct_elite_opps": "f1_pct_elite_opps"})
        )
        f2_sos = (
            sos_agg.filter(pl.col("fighter_role") == "f2")
            .drop("fighter_role")
            .rename({"avg_opp_elo": "f2_avg_opp_elo", "pct_elite_opps": "f2_pct_elite_opps"})
        )
        elo_df = (
            elo_df
            .join(f1_sos, on="root_fight_id", how="left")
            .join(f2_sos, on="root_fight_id", how="left")
            .with_columns([
                (pl.col("f1_avg_opp_elo") - pl.col("f2_avg_opp_elo"))
                .cast(f32).alias("avg_opp_elo_diff"),
            ])
        )

    return elo_df


ELITE_OPP_ELO_THRESHOLD = 1550.0   # threshold for "elite opponent" in SOS features

ERA_FIELD_SIZE      = 30   # top-N active fighters used to compute era benchmark
ERA_MIN_FIGHTS      = 5    # fighter must have this many UFC fights to count in the benchmark
PEAK_WINDOW_FIGHTS  = 5    # rolling N-fight window used to define a fighter's peak


def _compute_peak_elos(
    timeline: pl.DataFrame,
    layoff_days: int = 365,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (rolling_window_peak, era_adjusted_peak) per fighter.

    Rolling-window peak
    -------------------
    Mean pre-fight ELO over the best consecutive PEAK_WINDOW_FIGHTS-fight
    stretch within a contiguous activity window (no gap > layoff_days).

    Using a rolling average instead of a single-fight max fixes two biases:
      - Volume bias: Cerrone/Ferguson win many fights but their best 5-fight
        average is dragged down by losses to top competition mixed in. A
        fighter who beat mid-tier 20 times doesn't rank above one who
        dominated elite competition for 5 straight.
      - Short-career penalty: Khabib went 28-0; his best 5-fight stretch
        has a very high average. He's not punished for retiring young.
        Weidman's 2 Silva wins spike a single-fight peak but a 5-fight
        window around those includes weaker opposition, giving an honest avg.
      - Requires sustained dominance: you must string together N elite
        performances, not just one giant upset.

    Era-adjusted peak  (used for all-time display rankings)
    --------------------------------------------------------
    rolling_window_peak - mean_elo_of_top_{ERA_FIELD_SIZE}_active_fighters
                          at the date of the last fight in the peak window.

    Normalises for era inflation (modern roster is inflated) so cross-era
    comparisons are fairer.

    Known remaining limitation: division-depth bias (DJ, flyweights) is not
    fixed here — that requires weight-class-specific ELO systems.
    """
    elo: dict[str, float] = {}
    fight_count: dict[str, int] = {}     # total fights processed so far
    history: dict[str, list] = {}        # fighter_id -> [(date, pre_fight_elo)]
    all_events: list[tuple] = []         # (date, fighter_id, pre_fight_elo, fights_so_far)

    for row in timeline.iter_rows(named=True):
        f1_id  = row["fighter1_id"]
        f2_id  = row["fighter2_id"]
        winner = row["winner_id"]
        fd     = row["fight_date"]

        r1 = elo.get(f1_id, INITIAL_ELO)
        r2 = elo.get(f2_id, INITIAL_ELO)
        fc1 = fight_count.get(f1_id, 0)
        fc2 = fight_count.get(f2_id, 0)

        history.setdefault(f1_id, []).append((fd, r1))
        history.setdefault(f2_id, []).append((fd, r2))
        all_events.append((fd, f1_id, r1, fc1))
        all_events.append((fd, f2_id, r2, fc2))

        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        s1 = 1.0 if winner == f1_id else 0.0
        K1 = _get_k(fc1)
        K2 = _get_k(fc2)
        elo[f1_id] = r1 + K1 * (s1 - e1)
        elo[f2_id] = r2 + K2 * ((1.0 - s1) - (1.0 - e1))

        fight_count[f1_id] = fc1 + 1
        fight_count[f2_id] = fc2 + 1

    # ------------------------------------------------------------------
    # Build era benchmark: for each unique fight date, compute the mean
    # pre-fight ELO of the top-ERA_FIELD_SIZE fighters who have been
    # active (fought at least once) in the past layoff_days days.
    # ------------------------------------------------------------------
    all_events.sort(key=lambda x: x[0])
    unique_dates = sorted({e[0] for e in all_events})

    # latest_seen[fighter_id] = (date, elo, fights_at_that_point)
    latest_seen: dict[str, tuple] = {}
    event_idx = 0
    era_benchmark: dict = {}   # date -> float (None = pool too thin)

    for date in unique_dates:
        # Absorb all events on this date (pre-fight ELOs, no leakage)
        while event_idx < len(all_events) and all_events[event_idx][0] == date:
            _, fid, ev_elo, ev_fc = all_events[event_idx]
            if fid not in latest_seen or date >= latest_seen[fid][0]:
                latest_seen[fid] = (date, ev_elo, ev_fc)
            event_idx += 1

        # Top-ERA_FIELD_SIZE ELOs among experienced fighters active within layoff_days.
        # ERA_MIN_FIGHTS filter excludes rookies who are near 1500 and would
        # artificially deflate the benchmark — especially important in early years
        # when the active pool was small and mostly full of newcomers.
        active_elos = sorted(
            (ev_elo for fid, (ev_date, ev_elo, ev_fc) in latest_seen.items()
             if (date - ev_date).days <= layoff_days and ev_fc >= ERA_MIN_FIGHTS),
            reverse=True,
        )[:ERA_FIELD_SIZE]

        # Only compute a benchmark if we have a full ERA_FIELD_SIZE of
        # experienced fighters — avoids inflating early-era margins where
        # the qualified pool was tiny.
        era_benchmark[date] = (
            sum(active_elos) / len(active_elos)
            if len(active_elos) >= ERA_FIELD_SIZE
            else None
        )

    # ------------------------------------------------------------------
    # Compute per-fighter peaks using rolling PEAK_WINDOW_FIGHTS avg
    # ------------------------------------------------------------------
    raw_peaks: dict[str, float] = {}
    adjusted_peaks: dict[str, float] = {}

    for fid, hist in history.items():
        hist.sort(key=lambda x: x[0])

        # Split into contiguous activity windows (no gap > layoff_days)
        windows: list[list] = []
        cur: list = [hist[0]]
        for i in range(1, len(hist)):
            if (hist[i][0] - hist[i - 1][0]).days > layoff_days:
                windows.append(cur)
                cur = [hist[i]]
            else:
                cur.append(hist[i])
        windows.append(cur)

        # Find the best rolling-window avg across all activity windows.
        # Window size = min(PEAK_WINDOW_FIGHTS, fights in that window) so
        # short-career fighters aren't excluded — they just use all fights.
        best_avg  = -1.0
        best_date = None   # last fight date in best window (for era lookup)

        for w in windows:
            n = min(PEAK_WINDOW_FIGHTS, len(w))
            for i in range(len(w) - n + 1):
                chunk = w[i : i + n]
                avg   = sum(e for _, e in chunk) / n
                if avg > best_avg:
                    best_avg  = avg
                    best_date = chunk[-1][0]   # date of the window's last fight

        benchmark = era_benchmark.get(best_date) if best_date else None
        raw_peaks[fid]      = best_avg
        # None benchmark = era pool was too thin; exclude from display ranking
        adjusted_peaks[fid] = (best_avg - benchmark) if benchmark is not None else None

    return raw_peaks, adjusted_peaks


def print_top_fighters(fights: pl.DataFrame, prior_fights: pl.DataFrame, n: int = 10) -> None:
    """Print top-N fighters by final ELO rating."""
    timeline = _assemble_timeline(fights, prior_fights)
    final_elo, _ = _run_elo(timeline)

    names: dict[str, str] = {}
    for row in fights.select(["fighter1_id", "fighter1_name", "fighter2_id", "fighter2_name"]).iter_rows(named=True):
        names[row["fighter1_id"]] = row["fighter1_name"]
        names[row["fighter2_id"]] = row["fighter2_name"]

    ranked = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:n]

    print("\n%-6s %-25s %7s" % ("Rank", "Fighter", "ELO"))
    print("-" * 40)
    for i, (fid, rating) in enumerate(ranked, 1):
        print("%-6d %-25s %7.1f" % (i, names.get(fid, fid), rating))


def print_all_time_rankings(fights: pl.DataFrame, prior_fights: pl.DataFrame, n: int = 50) -> None:
    """Print top-N fighters ranked by era-adjusted peak ELO.

    Era-adjusted peak = peak_elo - mean_elo_of_top_30_active_fighters_at_peak_date.
    Answers "how far above the best available competition were you at your best?"
    Normalises for recency inflation, volume bias, and short-career penalty.
    """
    timeline = _assemble_timeline(fights, prior_fights)
    raw_peaks, adjusted_peaks = _compute_peak_elos(timeline)

    names: dict[str, str] = {}
    for row in fights.select(["fighter1_id", "fighter1_name", "fighter2_id", "fighter2_name"]).iter_rows(named=True):
        names[row["fighter1_id"]] = row["fighter1_name"]
        names[row["fighter2_id"]] = row["fighter2_name"]

    # Composite score = era_adj + 0.5 * (raw_peak - INITIAL_ELO)
    # Blends era dominance ("how far above your field") with absolute peak
    # quality ("how good were you in absolute terms"). Pure era-adj alone
    # can flip two fighters who peaked in slightly different eras despite
    # one clearly being better (e.g. Jones vs DC — Jones has the higher raw
    # peak and beat DC twice, but DC's era benchmark happened to be lower).
    valid = {
        fid: adj + 0.5 * (raw_peaks[fid] - INITIAL_ELO)
        for fid, adj in adjusted_peaks.items()
        if adj is not None
    }
    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:n]

    print("\n%-6s %-30s %16s %12s %10s" % ("Rank", "Fighter", "Peak (5-fight avg)", "+vs Era", "Score"))
    print("-" * 72)
    for i, (fid, score) in enumerate(ranked, 1):
        raw = raw_peaks.get(fid, 0.0)
        adj = adjusted_peaks.get(fid, 0.0)
        print("%-6d %-30s %16.1f %+12.1f %10.1f" % (i, names.get(fid, fid), raw, adj, score))


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
