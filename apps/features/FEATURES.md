# Feature Reference

> **Conventions:**
> - **(diff)** — single value computed as `f1 - f2`
> - **(f1 & f2)** — separate column per fighter with `f1_` / `f2_` prefix
> - All division-by-zero cases (`NaN`, `±inf`) are filled with `0.0` unless noted
> - "Career" = all prior fights before root fight; "Last 3" = 3 most recent; "Last fight" = most recent only
> - Aggregated rates use `sum(numerator) / sum(denominator)` across the window — never an average of per-fight rates

---

## Metadata (not features — identifiers / labels)

| Column | Notes |
|---|---|
| `meta_root_fight_id` | primary key |
| `meta_f1_id` | fighter 1 ID |
| `meta_f2_id` | fighter 2 ID |
| `meta_winner_id` | **target label** |
| `meta_loser_id` | |
| `meta_fight_date` | |
| `meta_end_time` | `MM:SS → seconds`; outcome duration of root fight |
| `meta_method` | raw finish method string |
| `meta_fight_type` | raw fight type string |

---

## Features — `fights_df`

### Fight Context

| Feature | Formula | Notes |
|---|---|---|
| `fight_format` | — | scheduled rounds (3 or 5) |
| `fight_type_id` | `FIGHT_TYPE_MAP` | 0=main, 1=title, 2=other; title implies extra pressure beyond round count |
| `weight_class_id` | `WEIGHT_CLASS_MAP` | 0–8; catch weight = 8 |

### Fighter Record **(f1 & f2)**

| Feature | Formula | Notes |
|---|---|---|
| `{p}_fight_count` | `prior_cnt` | total prior UFC fights; kept per-fighter — experience asymmetry matters |

### Stance **(f1 & f2)**

| Feature | Formula | Notes |
|---|---|---|
| `{p}_stance_id` | `STANCE_MAP` | 0=Orthodox, 1=Southpaw, 2=Switch, 3=Unknown; kept per-fighter — matchup (ortho vs. southpaw) is the signal, not the diff |

### Physical **(diff = f1 − f2)**

| Feature | Formula | Notes |
|---|---|---|
| `height_diff` | `f1_height_in − f2_height_in` | positive = f1 taller |
| `reach_diff` | `f1_reach_in − f2_reach_in` | positive = f1 longer reach |
| `f1_age` | `(fight_date − f1_dob).days / 365.25` | f1 age at fight time |
| `f2_age` | `(fight_date − f2_dob).days / 365.25` | f2 age at fight time |
| `age_diff` | `(f2_dob − f1_dob).days / 365.25` | positive = f1 older |

### Record **(diff)**

| Feature | Formula | Notes |
|---|---|---|
| `win_rate_diff` | `f1_win/(f1_win+f1_loss) − f2_win/(f2_win+f2_loss)` | from UFC snapshot; point-in-time, not leaky |

### ELO Rating

> Computed globally across all fights in chronological order. Pre-fight rating is recorded (no leakage). K=32, initial=1500.

| Feature | Formula | Notes |
|---|---|---|
| `f1_elo` | ELO rating of f1 before this fight | absolute quality signal |
| `f2_elo` | ELO rating of f2 before this fight | absolute quality signal |
| `elo_diff` | `f1_elo − f2_elo` | relative strength; main predictive signal |

---

## Features — `prior_fights_df`

> All aggregated per `(root_fight_id, fighter_role)`. Only fights **before** the root fight date.

### Activity **(f1 & f2)**

| Feature | Formula | Notes |
|---|---|---|
| `{p}_years_since_last_fight` | `(root_fight_date − max(fight_date)).days / 365.25` | layoff length; rust indicator |
| `{p}_avg_fights_per_year` | `total_fights / career_years` | career activity pace |
| `{p}_fights_this_year` | count where `fight_date >= root − 365d` | recent activity |
| `{p}_fights_last_3yrs` | count where `fight_date >= root − 1095d` | medium-term activity |

### Method Rates — Career **(f1 & f2)**

> `dec_win_rate` and `dec_loss_rate` excluded — they equal `1 − ko_rate − sub_rate` (perfect multicollinearity).

| Feature | Formula | Notes |
|---|---|---|
| `{p}_ko_win_rate` | `ko_wins / total_wins` | finishing power via KO/TKO |
| `{p}_sub_win_rate` | `sub_wins / total_wins` | submission offense |
| `{p}_ko_loss_rate` | `ko_losses / total_losses` | KO vulnerability |
| `{p}_sub_loss_rate` | `sub_losses / total_losses` | submission vulnerability |

### Weight Class Stats **(f1 & f2)**

> Filtered to same weight class as root fight. `wins_at_weight` and `losses_at_weight` excluded — derivable from `fights_at_weight` + method breakdown.

| Feature | Formula | Notes |
|---|---|---|
| `{p}_fights_at_weight` | count where `weight_class == root_weight_class` | experience at this weight |
| `{p}_ko_wins_at_weight` | KO/TKO wins at this weight | |
| `{p}_sub_wins_at_weight` | submission wins at this weight | |
| `{p}_dec_wins_at_weight` | decision wins at this weight | |
| `{p}_ko_losses_at_weight` | KO/TKO losses at this weight | |
| `{p}_sub_losses_at_weight` | submission losses at this weight | |
| `{p}_dec_losses_at_weight` | decision losses at this weight | |

### Round Format Experience **(f1 & f2)**

> `3rd_fights` excluded — approximately equals `fight_count − 5rd_fights`.

| Feature | Formula | Notes |
|---|---|---|
| `{p}_3rd_wins` | wins in 3-round fights | |
| `{p}_3rd_losses` | losses in 3-round fights | |
| `{p}_5rd_fights` | total 5-round fights | championship/main event experience |
| `{p}_5rd_wins` | wins in 5-round fights | |
| `{p}_5rd_losses` | losses in 5-round fights | |

### Fight Duration **(f1 & f2)**

> Kept per-fighter (not diffed) — absolute fight length tendency is a meaningful individual trait.

| Feature | Formula | Notes |
|---|---|---|
| `{p}_last_fight_end_time_s` | `end_time_s` of most recent fight | recency-weighted finish time |
| `{p}_last_3_avg_end_time_s` | `mean(end_time_s)` of last 3 fights | short-term pace |
| `{p}_avg_end_time_s` | `mean(end_time_s)` career | career finish tendency; high = decision fighter |
| `{p}_total_time_fought_s` | `sum(end_time_s)` career | total seconds spent in the cage; complements fight_count with volume-of-action info |

### Momentum **(f1 & f2)**

> Kept per-fighter — both fighters' streaks are visible to the model simultaneously.

| Feature | Formula | Notes |
|---|---|---|
| `{p}_win_streak` | consecutive wins from most recent fight | current hot streak; 0 if most recent is a loss |
| `{p}_loss_streak` | consecutive losses from most recent fight | current cold streak; 0 if most recent is a win |

### Striking & Grappling Stats **(diff)**

> Three windows × 8 stats = 24 diffs. Plus 8 trend diffs.

| Feature | Formula | Notes |
|---|---|---|
| `last_fight_slpm_diff` | `sig_str_landed / (end_time_s / 60)` — most recent fight | |
| `last_3_slpm_diff` | same, summed over last 3 | |
| `slpm_diff` | career | sig strikes landed per minute |
| `last_fight_str_acc_diff` | `sig_str_landed / sig_str_attempts` — most recent | |
| `last_3_str_acc_diff` | | |
| `str_acc_diff` | career | striking accuracy |
| `last_fight_sapm_diff` | `opp_sig_str_landed / (end_time_s / 60)` — most recent | strikes absorbed per minute |
| `last_3_sapm_diff` | | |
| `sapm_diff` | career | |
| `last_fight_str_def_diff` | `1 − opp_sig_str_landed / opp_sig_str_attempts` — most recent | strike defense rate |
| `last_3_str_def_diff` | | |
| `str_def_diff` | career | |
| `last_fight_td_avg_diff` | `td_landed / (end_time_s / 60) * 15` — most recent | takedowns per 15 min |
| `last_3_td_avg_diff` | | |
| `td_avg_diff` | career | |
| `last_fight_td_acc_diff` | `td_landed / td_attempts` — most recent | takedown accuracy |
| `last_3_td_acc_diff` | | |
| `td_acc_diff` | career | |
| `last_fight_td_def_diff` | `1 − opp_td_landed / opp_td_attempts` — most recent | takedown defense rate |
| `last_3_td_def_diff` | | |
| `td_def_diff` | career | |
| `last_fight_sub_avg_diff` | `sub_att / (end_time_s / 60) * 15` — most recent | sub attempts per 15 min |
| `last_3_sub_avg_diff` | | |
| `sub_avg_diff` | career | |

**Trend ratios** — `last_3 / career` per stat, clipped [0, 3], neutral fill 1.0 when career = 0

| Feature | Formula | Notes |
|---|---|---|
| `slpm_trend_diff` | `(f1_last_3_slpm / f1_slpm) − (f2_last_3_slpm / f2_slpm)` | > 0 = f1 output trending up vs f2 |
| `str_acc_trend_diff` | same pattern | > 0 = f1 getting more accurate recently |
| `sapm_trend_diff` | | > 0 = f1 absorbing more recently (declining defense) |
| `str_def_trend_diff` | | |
| `td_avg_trend_diff` | | |
| `td_acc_trend_diff` | | |
| `td_def_trend_diff` | | |
| `sub_avg_trend_diff` | | |

### Advanced Stats **(diff)**

> `NaN` and `±inf` handled via `when(is_nan | is_infinite).then(0.0)`.

**KD rate** — 3 windows

| Feature | Formula | Notes |
|---|---|---|
| `last_fight_kd_rate_diff` | `kd / sig_str_landed` — most recent fight | one-punch power; most predictive window |
| `last_3_kd_rate_diff` | summed over last 3 | |
| `kd_rate_diff` | career | |

**Net control pct** — 2 windows

| Feature | Formula | Notes |
|---|---|---|
| `last_3_net_ctrl_pct_diff` | `(ctrl_time_s − opp_ctrl_time_s) / end_time_s` over last 3 | grappling dominance; +1 = all control, −1 = all controlled |
| `net_ctrl_pct_diff` | career | |

**Sig-to-total ratio** — 2 windows

| Feature | Formula | Notes |
|---|---|---|
| `last_3_sig_to_total_ratio_diff` | `sig_str_landed / total_str_landed` over last 3 | purposeful vs. volume striking; high = efficient |
| `sig_to_total_ratio_diff` | career | |

**Strike zone profile** — career only

> `head_str_pct` excluded (head + body + leg = 1.0); `distance_str_pct` excluded (dist + clinch + ground = 1.0).

| Feature | Formula | Notes |
|---|---|---|
| `body_str_pct_diff` | `body_landed / sig_str_landed` | body attack tendency |
| `leg_str_pct_diff` | `leg_landed / sig_str_landed` | leg kick game |
| `clinch_str_pct_diff` | `clinch_landed / sig_str_landed` | clinch striking tendency |
| `ground_str_pct_diff` | `ground_landed / sig_str_landed` | ground striking tendency |

**Grappling & durability** — career only

| Feature | Formula | Notes |
|---|---|---|
| `gnp_rate_diff` | `ground_landed / (ctrl_time_s / 60)` | GnP vs. just holding; higher = active on top |
| `chin_score_diff` | `opp_kd / opp_sig_str_landed` | higher = more fragile chin |
| `reversal_rate_diff` | `rev / (opp_ctrl_time_s / 60)` | scramble / escape ability per minute controlled |
| `def_sub_exposure_diff` | `opp_sub_att / opp_td_landed` | sub vulnerability when taken down |

---

## Features — `prior_rounds_df` **(diff)**

> Round-by-round stats split into "own" (fighter's rows) and "opponent" (opponent's rows) per prior fight,
> identified via `opponent_id` join with `prior_fights_df`. Early = rounds 1–2, Late = rounds 3+.
> `NaN` / `±inf` filled with `0.0`; pace ratio `0/0` filled with `1.0` (neutral — no late rounds fought).

**Career windows**

| Feature | Formula | Notes |
|---|---|---|
| `r1_sig_per_fight_diff` | `sum(r1 sig_str_landed) / count(r1 rounds)` | avg R1 output per fight — early aggression |
| `r1_kd_rate_diff` | `sum(r1 kd) / sum(r1 sig_str_landed)` | R1 finishing power; KOs cluster in R1 |
| `slpm_pace_ratio_diff` | `(late_sig/late_rds) / (early_sig/early_rds)` | < 1 = fades late; cardio signal |
| `sapm_pace_ratio_diff` | `(opp_late_sig/opp_late_rds) / (opp_early_sig/opp_early_rds)` | > 1 = opponent lands more late; defense breaks down |
| `str_acc_degradation_diff` | `(late_sig/late_sig_att) / (early_sig/early_sig_att)` | accuracy under fatigue; < 1 = gets sloppy late |
| `body_escalation_diff` | `(late_body/late_rds) / (early_body/early_rds)` | ramps up body work in later rounds — tactical signal |
| `late_ctrl_per_fight_diff` | `sum(ctrl_time_s in r3+) / fights_with_late_rounds` | grappling endurance — avg seconds of control in late rounds |
| `late_td_acc_diff` | `sum(r3+ td_landed) / sum(r3+ td_attempts)` | takedown precision when tired |
| `late_sub_per_round_diff` | `sum(r3+ sub_att) / count(r3+ rounds)` | submission threat when opponent is tired |
| `rd_dom_rate_diff` | `mean(own_sig > opp_sig per round)` | % of rounds where they out-struck opponent; 0.5 = neutral |
| `last_rd_str_diff_diff` | `mean(own_sig − opp_sig in final round of each fight)` | last-round output advantage; clutch / championship pacing |
| `post_kd_response_diff` | `(avg own_sig in rounds after taking KD) / career_avg_sig_per_round` | > 1 = bounces back strong; < 1 = turtles up after being dropped |

**Last 3 fights windows**

| Feature | Formula | Notes |
|---|---|---|
| `last_3_r1_sig_per_fight_diff` | same as career, last 3 fights only | recent R1 aggression |
| `last_3_slpm_pace_ratio_diff` | same, last 3 | recent cardio trend |
| `last_3_sapm_pace_ratio_diff` | same, last 3 | recent defense durability trend |
| `last_3_str_acc_degradation_diff` | same, last 3 | recent accuracy-under-fatigue trend |
| `last_3_rd_dom_rate_diff` | same, last 3 | recent round dominance |
| `last_3_last_rd_str_diff_diff` | same, last 3 | recent last-round performance |
| `last_3_post_kd_response_diff` | same, last 3 | recent response to adversity |
