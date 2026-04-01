# ELO System Design

## Overview

The ELO system serves two distinct purposes with different requirements:

| Purpose | What we want | Key concern |
|---|---|---|
| **ML feature** | Fighter's true quality at fight time | No leakage, point-in-time accuracy |
| **UI / All-time rankings** | Career greatness, human-readable | Recency bias, era fairness, decay |

These two use cases pull in opposite directions. A fighter who peaked in 2012 and declined after should have a low current ELO (correct for ML) but rank highly all-time (correct for display). We handle them separately.

---

## Current Implementation

**Variable K schedule** — tuned via bucket-MAE calibration (7.5x improvement over fixed K=32):

| Fight count | K | Rationale |
|---|---|---|
| 0–4 | 66 | Rookie — high uncertainty, rating should move fast |
| 5–14 | 68 | Developmental — results highly informative as fighter establishes level |
| 15–19 | 63 | Settling — true level becoming clear |
| 20+ | 54 | Veteran — stable career, rating shouldn't swing on single fight |

**Calibration loss:** 0.0049 (vs 0.0366 for standard fixed K=32)

**Stat blending rejected** — binary win/loss only (alpha=0). Tested blending sig strikes, ctrl time, KDs, and TDs into the score but it consistently hurt calibration. UFC upsets are frequent enough that the result IS the ground truth — a fighter who dominated stats but got KO'd by a lucky punch still lost.

**Timeline construction** — processes ALL fights chronologically (root fights + prior fight history arrays), not just root fights in isolation. This ensures a fighter's ELO at fight time reflects their full career through that date, including early bouts that only appear as embedded history.

---

## Known Limitations

### 1. ELO Inflation Over Time
New fighters enter at 1500. When they lose to established fighters, they pump ELO upward. Active fighters accumulate from this inflated pool. GSP's 1702 was earned in 2013 — worth more then than now, but the number doesn't reflect era context.

**Symptom:** Our top 25 skews heavily toward active (2020+) fighters. Legends who retired early appear underrated.

**Fix for UI:** Era-relative ELO — `fighter_elo - rolling_avg_elo_of_all_active_fighters_that_year`. Shows how far above peers they were, normalized for era.

**Fix for ML:** Not needed — we use `elo_diff` between two fighters in the same fight, same era. The diff is accurate even if the absolute numbers are inflated.

### 2. Late-Career Losses Tank Legends
Anderson Silva (current ELO: 1553) was arguably the greatest fighter ever at his peak, but Weidman, Romero, and a string of late-career losses bled his ELO back into the pool. BJ Penn (1422) and Matt Hughes (1464) have the same problem.

**Fix:** Peak ELO — see below.

### 3. Non-UFC Data Missing
Fedor Emelianenko and Wanderlei Silva built their legacies primarily in PRIDE FC. Our dataset is UFC-only, so they enter at 1500 on their first UFC appearance regardless of their actual standing. Their ratings are not trustworthy.

**Fix:** Cannot fix without non-UFC fight data. Flag these fighters in UI as "ELO unreliable — significant non-UFC career."

### 4. Cross-Division ELO Mixing
A win over a UFC heavyweight earns the same ELO as a win over a flyweight. Talent pool depth varies significantly by division. Jon Jones' LHW ELO gets mixed with his HW ELO when he changes weight class, which distorts both.

**Fix:** Weight class-specific ELO — see below.

---

## Planned Improvements

### Peak ELO (All-Time Rankings)

For display purposes, rank fighters by their **peak ELO** — the highest rating they achieved at any point in their career, with a small decline window to smooth out single-fight noise.

```
peak_elo = max(elo[t]) over all fight times t
peak_elo_smoothed = max(rolling_3_fight_avg_elo)  # avoids a fluky single win distorting peak
```

This correctly places Anderson Silva near the top (his peak was ~1800+) while his final rating (1553) reflects his actual current level.

**For the ML feature:** use point-in-time ELO, not peak ELO. You want to know how good they are now, not how good they were at their best. But `peak_elo` and `elo_vs_peak` (= `current_elo / peak_elo`) are valuable additional features — they capture whether a fighter is ascending, at their ceiling, or in decline.

### Weight Class-Specific ELO

Run independent ELO systems per division. For each fight, only update the weight class ELO corresponding to the fight's division.

```
fighter.elo_lhw  # light heavyweight ELO
fighter.elo_ww   # welterweight ELO
fighter.elo_lw   # lightweight ELO
```

When a fighter moves divisions (Jones LHW → HW), their HW ELO starts at their LHW ELO as a prior — they've demonstrated high overall quality. Decay applies to the inactive division.

**Benefits:**
- GSP's welterweight dominance is reflected in WW ELO without being diluted by fighters from other divisions
- Division-specific ELO diff becomes a feature: `elo_diff_at_weight` is more precise than global `elo_diff`
- Era inflation is contained within divisions — less mixing of talent pool eras

### Era-Relative ELO (UI Display Only)

For ranking legends fairly across eras:

```
era_relative_elo = fighter_elo_at_peak - avg_elo_of_top_20_fighters_that_year
```

Measures how far above the competition you were at your best, not your absolute ELO number. GSP dominating WW in 2012 when the era average was 1600 is correctly shown as more impressive than a modern fighter sitting 100 points above a 1700 era average.

---

## ELO Decay

### For ML Features — Likely Unnecessary

The model already has `years_since_last_fight` and `fights_last_3yrs` as explicit features. XGBoost will learn the interaction between layoff length and performance degradation from those features without needing ELO decay baked in. Decaying ELO for ML would double-count the same signal and could introduce noise.

**Verdict:** Skip ELO decay for ML features. Trust the model to learn rust from the explicit activity features.

### For UI / Dashboard — Recommended

Users visiting the dashboard during a fighter's long layoff should see their rating naturally slip below newly active stars. A fighter who hasn't fought in 18 months while others have been active and climbing should reflect that reality.

**Proposed decay formula:**
```
days_inactive = (today - last_fight_date).days
if days_inactive > 180:
    decay_rate = 0.15 * K_veteran  # ~8 ELO points/month after 6mo
    decayed_elo = current_elo - (decay_rate * (days_inactive - 180) / 30)
    displayed_elo = max(decayed_elo, 1500)  # floor at baseline
```

This means:
- 6 months inactive: no decay (normal camp/recovery)
- 12 months inactive: ~48 points decay
- 24 months inactive: ~192 points decay

Keeps active fighters appropriately ahead of long-inactive ones in the public display without affecting the stored historical snapshots used for ML.

---

## Additional Rating Systems (Planned)

These run parallel to the overall ELO and serve as both better ELO inputs and standalone ML features:

### Striking Offense / Defense Rating
- **Offensive:** ELO updated based on sig_str_landed relative to opponent's defensive rating
- **Defensive:** ELO updated based on sig_str_absorbed relative to opponent's offensive rating
- A win over a fighter with high defensive rating earns more offensive ELO than the same result against a poor defender

### Grappling Offense / Defense Rating
- **Offensive:** TD landed + ctrl time + sub attempts, weighted by opponent's grappling defense rating
- **Defensive:** TD stuffing rate + escape rate, weighted by opponent's grappling offense rating

### Finishing / Durability Rating
- **Finishing:** Finish rate weighted by opponent's durability rating — stopping a fighter with high durability earns more
- **Durability:** Inverse — how often opponents finish you vs how often they'd be expected to based on their finishing rating

### Momentum / Peak Form Rating (Decay-First System)
- Same ELO but older fights decay exponentially
- Captures current form rather than career average
- GSP in retirement drops. Chimaev on a 6-fight finish streak peaks.
- Best used as a feature alongside standard ELO — model can learn "current ELO vs peak ELO" signal

---

## Feature Summary

| Feature | Source | Use |
|---|---|---|
| `f1_elo`, `f2_elo`, `elo_diff` | Standard ELO at fight time | Current ML feature (implemented) |
| `f1_peak_elo`, `f2_peak_elo` | Max ELO reached before fight date | Is this their peak or are they past it? |
| `f1_elo_vs_peak`, `f2_elo_vs_peak` | `current / peak` | Ascending (>1.0) vs declining (<1.0) |
| `f1_elo_at_weight`, `elo_diff_at_weight` | Division-specific ELO | More precise than global ELO |
| `f1_striking_off_elo`, `f2_striking_def_elo` | Style-specific ELO | Explicit style matchup signal |
| `f1_grappling_off_elo`, `f2_grappling_def_elo` | Style-specific ELO | Explicit style matchup signal |
| `f1_finishing_elo`, `f2_durability_elo` | Style-specific ELO | Finish probability signal |
| `f1_momentum_elo`, `elo_diff_momentum` | Decay-weighted ELO | Current hot/cold streak signal |
