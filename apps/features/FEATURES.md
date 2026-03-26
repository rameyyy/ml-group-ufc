# Feature Reference

> **Convention:** Columns marked **(f1 & f2)** exist for both fighters with the respective prefix, e.g. `f1_age` and `f2_age`.

---

## Metadata (not features — identifiers / labels)

| Column | Source | Notes |
|---|---|---|
| `meta_root_fight_id` | `fights_df` | primary key |
| `meta_f1_id` | `fights_df` | |
| `meta_f2_id` | `fights_df` | |
| `meta_winner_id` | `fights_df` | **target label** |
| `meta_loser_id` | `fights_df` | |
| `meta_fight_date` | `fights_df` | |
| `meta_end_time` | `fights_df` | seconds, derived from `end_time` string |
| `meta_method` | `fights_df` | raw string |
| `meta_fight_type` | `fights_df` | raw string |

---

## Features — `fights_df`

### Fight Context

| Column | Type | Notes |
|---|---|---|
| `fight_format` | Int16 | number of scheduled rounds |
| `fight_type_id` | Int8 | mapped via `FIGHT_TYPE_MAP` |
| `weight_class_id` | Int8 | mapped via `WEIGHT_CLASS_MAP` |
| `referee_id` | Int16 | mapped via `REFEREE_MAP` |

### Physical **(f1 & f2)**

| Feature | Type | Notes |
|---|---|---|
| `{p}_height_in` | | |
| `{p}_weight_lbs` | | |
| `{p}_reach_in` | | |
| `{p}_stance_id` | Int8 | mapped via `STANCE_MAP` |
| `{p}_age` | Float32 | `(fight_date - dob).days / 365.25` |

### Record **(f1 & f2)**

| Feature | Type | Notes |
|---|---|---|
| `{p}_fight_count` | u32 | total prior fights |
| `{p}_win` | | |
| `{p}_loss` | | |
| `{p}_win_rate` | Float32 | `win / (win + loss)` |

### ⚠️ UFC Stats Page Aggregates — Potential Leakage **(f1 & f2)**

> Running career averages from the UFC stats page at snapshot time.
> Include data from **all fights up to and including the root fight**.
> **May need to be dropped and recalculated from `prior_fights_df`.**

| Feature | Notes |
|---|---|
| `{p}_slpm` | sig strikes landed per minute |
| `{p}_str_acc` | striking accuracy |
| `{p}_sapm` | sig strikes absorbed per minute |
| `{p}_str_def` | strike defense |
| `{p}_td_avg` | takedowns per 15 min |
| `{p}_td_acc` | takedown accuracy |
| `{p}_td_def` | takedown defense |
| `{p}_sub_avg` | sub attempts per 15 min |

---

## Features — `prior_fights_df`

> Aggregated per `root_fight_id` + `fighter_role`. All clean — only fights **before** the root fight.

### Activity **(f1 & f2)**

| Feature | Notes |
|---|---|
| `{p}_years_since_last_fight` | Float32, years since most recent prior fight |
| `{p}_avg_fights_per_year` | career fight frequency |
| `{p}_fights_this_year` | fights in rolling 365-day window before root fight |
| `{p}_fights_last_2yrs` | fights in rolling 730-day window |
| `{p}_fights_last_3yrs` | fights in rolling 1095-day window |

### Method Counts — Career **(f1 & f2)**

| Feature | Notes |
|---|---|
| `{p}_ko_wins` | career KO/TKO wins |
| `{p}_sub_wins` | career submission wins |
| `{p}_dec_wins` | career decision wins (unan + maj + split) |
| `{p}_ko_losses` | career KO/TKO losses |
| `{p}_sub_losses` | career submission losses |
| `{p}_dec_losses` | career decision losses |

### Weight Class Stats **(f1 & f2)**

> Filtered to the same weight class as the root fight.

| Feature | Notes |
|---|---|
| `{p}_fights_at_weight` | total fights at this weight class |
| `{p}_wins_at_weight` | |
| `{p}_losses_at_weight` | |
| `{p}_ko_wins_at_weight` | |
| `{p}_sub_wins_at_weight` | |
| `{p}_dec_wins_at_weight` | |
| `{p}_ko_losses_at_weight` | |
| `{p}_sub_losses_at_weight` | |
| `{p}_dec_losses_at_weight` | |

### Round Format Experience **(f1 & f2)**

| Feature | Notes |
|---|---|
| `{p}_3rd_fights` | prior fights scheduled for 3 rounds |
| `{p}_3rd_wins` | wins in 3-round fights |
| `{p}_3rd_losses` | losses in 3-round fights |
| `{p}_5rd_fights` | prior fights scheduled for 5 rounds |
| `{p}_5rd_wins` | wins in 5-round fights |
| `{p}_5rd_losses` | losses in 5-round fights |

### Fight Duration **(f1 & f2)**

| Feature | Notes |
|---|---|
| `{p}_last_fight_end_time_s` | most recent fight end time in seconds |
| `{p}_last_3_avg_end_time_s` | avg end time (seconds) across last 3 fights |
| `{p}_avg_end_time_s` | avg end time (seconds) across all prior fights |
| `{p}_last_fight_rounds` | `last_fight_end_time_s / 300` |
| `{p}_last_3_avg_rounds` | `last_3_avg_end_time_s / 300` |
| `{p}_avg_rounds` | `avg_end_time_s / 300` |

---

## Features — `prior_rounds_df` (planned)

> Per-round granularity. Enables pace-normalized and late-round stats.

| Feature | Notes |
|---|---|
| `{p}_avg_sig_str_per_round` | normalized to round pace |
| `{p}_avg_td_per_round` | |
| `{p}_late_round_sig_str` | avg sig str in round 3+ — cardio proxy |
| `{p}_early_round_sig_str` | avg sig str in rounds 1-2 — early aggression |
| `{p}_late_vs_early_str_ratio` | pace drop-off indicator |
