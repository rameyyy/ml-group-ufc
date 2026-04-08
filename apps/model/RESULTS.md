# Model Results & Development Log

## Final Results (Current)

| Metric | Value |
|---|---|
| Test accuracy | **76.3%** |
| Train accuracy | 76.0% |
| Overfit gap | **-0.3pp** (zero overfitting) |
| Test AUC | **0.8302** |
| Test log-loss | **0.5234** |
| Best iteration | 1,010 / 2,000 |
| Features | 200 |
| Train rows | 8,172 (post mirror-augment) |
| Test rows | 1,022 |

**Split:** Chronological 80/20 — oldest 80% of fights train, most recent 20% test. No shuffling, zero future leakage.

---

## Progress Over Time

| Phase | Test Acc | Overfit Gap | AUC | Features | Notes |
|---|---|---|---|---|---|
| Baseline | 72.4% | 11.8pp | 0.818 | 125 | Raw stats, fixed K=32 ELO |
| + ELO features | ~72.1% | ~11.5pp | ~0.816 | ~130 | Variable K ELO, peak ELO, division ELO |
| + Feature expansion | 74.9% | 4.1pp | 0.827 | 198 | Domain ELOs, velocity, SOS, damage, tendency, recency-weighted stats, matchup interactions |
| + defensive_ctrl_pct | 74.8% | 6.5pp | 0.823 | 200 | Positional pressure proxy |
| + Optuna tuning | 76.2% | -0.3pp | 0.830 | 200 | 75 trials, TimeSeriesSplit CV |
| + Retrain to convergence | **76.3%** | **-0.3pp** | **0.830** | 200 | Early stopping found true ceiling at 1,010 trees |

---

## Optimised Hyperparameters

Found via Optuna (75 trials, TimeSeriesSplit 5-fold CV, optimising AUC).

```python
max_depth           = 3
learning_rate       = 0.01534
subsample           = 0.5376
colsample_bytree    = 0.8829
min_child_weight    = 3
gamma               = 2.7404
reg_alpha           = 2.7040
reg_lambda          = 3.2899
n_estimators        = 2000   # early stopping triggers at ~1010
early_stopping_rounds = 30
```

**Key finding:** Optuna found much heavier regularisation than the manually-guessed parameters in MODEL.md — especially `reg_alpha` (0.1 → 2.70) and `reg_lambda` (1.0 → 3.29). Learning rate also shifted from 0.05 to 0.015, requiring more trees to converge. This confirms the manual guesses were in the right direction but well short of the true optimum.

---

## Architecture

### Train / Test Split
Chronological 80/20. All 5,185 fights sorted by date; oldest 4,086 train, most recent 1,022 test.

### Mirror Augmentation
Training set doubled by swapping f1/f2 labels on every fight — forces the model to treat both roles symmetrically. Raw train win rate was 56% f1 (data artifact from UFC's assignment convention shifting over time). Post-augment: exactly 50/50. Applied **per fold** during CV to prevent leakage into validation sets.

### ELO System
Variable K schedule tuned via bucket-MAE calibration (7.5x improvement over fixed K=32):

| Fight count | K |
|---|---|
| 0–4 | 66 |
| 5–14 | 68 |
| 15–19 | 63 |
| 20+ | 54 |

Per-division ELO with dynamic transfer factors calibrated via held-out accuracy backtest. Division ELO uses division fight count for K (not global), so new entrants adjust quickly.

---

## Feature Set (200 total)

### Fight Context (4)
`fight_format`, `fight_type_id`, `weight_class_id`, `southpaw_advantage`

### Fighter Record (10)
`f1/f2_fight_count`, `f1/f2_stance_id`, `height_diff`, `reach_diff`, `f1/f2_age`, `age_diff`, `win_rate_diff`

### Activity (8)
`f1/f2_years_since_last_fight`, `f1/f2_avg_fights_per_year`, `f1/f2_fights_this_year`, `f1/f2_fights_last_3yrs`

### Method Rates (8)
`f1/f2_ko_win_rate`, `f1/f2_sub_win_rate`, `f1/f2_ko_loss_rate`, `f1/f2_sub_loss_rate`

### At-Weight Records (14)
Fights, KO/sub/dec wins and losses at root fight's weight class — per fighter.

### Fight Duration (8)
Last fight end time, last-3 avg, career avg, total time fought — per fighter.

### Format Experience (10)
3-round wins/losses, 5-round fight count/wins/losses — per fighter.

### Streaks (4)
`f1/f2_win_streak`, `f1/f2_loss_streak`

### Damage Accumulated (12)
Career sig strikes absorbed, head strikes absorbed, knockdowns absorbed, KO losses, KO losses in last 3 fights, fights since last KO loss — per fighter.

### Fight Tendency (6)
Pct fights to decision, pct fights finished by fighter, pct fights finished by opponent — per fighter.

### ELO System (34)
| Feature | Description |
|---|---|
| `f1/f2_elo` | Overall ELO at fight time |
| `f1/f2_peak_elo` | Peak ELO (rolling 5-fight window, resets on 2-loss streak or 365d layoff) |
| `f1/f2_elo_vs_peak` | Current ELO / peak ELO — ascending vs declining signal |
| `f1/f2_elo_at_weight` | Division-specific ELO |
| `f1/f2_str_off/def_elo` | Striking offense / defense ELO |
| `f1/f2_grap_off/def_elo` | Grappling offense / defense ELO |
| `f1/f2_finish_elo` | Finishing rate ELO |
| `f1/f2_durability_elo` | Durability (resist finishing) ELO |
| `f1/f2_elo_delta_last_3/5` | ELO change over last 3 / 5 fights |
| `f1/f2_avg_opp_elo` | Mean ELO of all prior opponents at fight time (strength of schedule) |
| `f1/f2_pct_elite_opps` | Pct of prior opponents with ELO > 1550 |
| `elo_diff` | f1 - f2 overall ELO |
| `elo_diff_at_weight` | f1 - f2 division ELO |
| `str_matchup_diff` | (f1_str_off - f2_str_def) - (f2_str_off - f1_str_def) |
| `grap_matchup_diff` | Same pattern for grappling |
| `elo_delta_diff_last_3` | f1 velocity - f2 velocity (last 3 fights) |
| `avg_opp_elo_diff` | f1 SOS - f2 SOS |

### Striking Stats (32)
8 core stats (slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg) × last fight / last 3 / career windows, all in diff form (f1 - f2). Plus trend ratios (last_3 / career).

### Advanced Stats (15)
KD rate (3 windows), net ctrl pct (2 windows), defensive ctrl pct (2 windows), sig-to-total ratio, body/leg/clinch/ground str pct, GnP rate, chin score, reversal rate, defensive sub exposure — all diff form.

### Round-by-Round (19)
R1 output, pace ratios (early vs late), str accuracy degradation, body escalation, late-round ctrl/TD/sub, round dominance rate, last-round str diff, post-KD response — career and last-3 windows, all diff form.

### Recency-Weighted Stats (8)
Exponentially decay-weighted versions of 8 core striking/grappling stats (half-life = 3 fights), diff form.

### Matchup Interactions (6)
`ko_rate_diff`, `sub_rate_diff`, `finish_matchup_diff`, `decision_tendency_diff`, `f1_finish_rate_vs_f2_durability`, `f2_finish_rate_vs_f1_durability`

---

## Top 50 Features by Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | win_rate_diff | 0.0222 |
| 2 | str_matchup_diff | 0.0128 |
| 3 | rd_dom_rate_diff | 0.0102 |
| 4 | age_diff | 0.0097 |
| 5 | finish_matchup_diff | 0.0093 |
| 6 | last_rd_str_diff_diff | 0.0090 |
| 7 | f2_grap_def_elo | 0.0089 |
| 8 | f2_fights_last_3yrs | 0.0089 |
| 9 | f1_fights_last_3yrs | 0.0085 |
| 10 | f2_total_time_fought_s | 0.0084 |
| 11 | f1_avg_fights_per_year | 0.0081 |
| 12 | f2_avg_fights_per_year | 0.0081 |
| 13 | f1_fights_this_year | 0.0078 |
| 14 | f2_sub_wins_at_weight | 0.0075 |
| 15 | f1_total_time_fought_s | 0.0074 |
| 16 | f1_fight_count | 0.0072 |
| 17 | f1_str_off_elo | 0.0070 |
| 18 | f2_fight_count | 0.0070 |
| 19 | f1_3rd_wins | 0.0069 |
| 20 | f2_win_streak | 0.0069 |
| 21 | f2_durability_elo | 0.0067 |
| 22 | f2_ko_losses_at_weight | 0.0066 |
| 23 | str_def_diff | 0.0064 |
| 24 | f1_sub_win_rate | 0.0063 |
| 25 | f1_years_since_last_fight | 0.0063 |
| 26 | f2_str_off_elo | 0.0062 |
| 27 | last_3_last_rd_str_diff_diff | 0.0061 |
| 28 | late_td_acc_diff | 0.0061 |
| 29 | f1_fights_at_weight | 0.0061 |
| 30 | f1_3rd_losses | 0.0060 |
| 31 | kd_rate_diff | 0.0060 |
| 32 | f2_5rd_wins | 0.0058 |
| 33 | f1_durability_elo | 0.0058 |
| 34 | f2_5rd_fights | 0.0058 |
| 35 | f2_fights_at_weight | 0.0057 |
| 36 | f1_pct_fights_finished | 0.0057 |
| 37 | f1_5rd_losses | 0.0057 |
| 38 | last_3_td_def_diff | 0.0057 |
| 39 | f2_str_def_elo | 0.0056 |
| 40 | rw_td_avg_diff | 0.0056 |
| 41 | sub_avg_diff | 0.0056 |
| 42 | late_sub_per_round_diff | 0.0056 |
| 43 | f1_career_sig_str_absorbed | 0.0056 |
| 44 | f1_5rd_fights | 0.0056 |
| 45 | f2_years_since_last_fight | 0.0056 |
| 46 | f2_3rd_losses | 0.0056 |
| 47 | f2_pct_elite_opps | 0.0055 |
| 48 | f2_age | 0.0055 |
| 49 | last_3_rd_dom_rate_diff | 0.0055 |
| 50 | f1_age | 0.0055 |

---

## Remaining Roadmap

### Features Not Yet Built
| Feature | Description | Expected value |
|---|---|---|
| Wrestler vs striker index | Continuous grappler-to-striker score from td_avg + ctrl_time vs slpm + str_acc | High — explicit style matchup |
| Performance vs elite | slpm/str_acc specifically vs opponents with ELO > 1600 | Medium — levels up or gets exposed |
| Takedown-to-sub conversion | sub_att / td_landed — once on the mat, how dangerous? | Medium |
| Big fight performance | Stats in 5-round fights specifically | Medium |
| Non-UFC fight history | PRIDE/Bellator records — fixes cold start for legacy fighters | High if obtainable |

### Architecture
| Step | Description | Expected gain |
|---|---|---|
| Ensemble | XGBoost + LightGBM + CatBoost averaged probabilities | +0.5–1.5% AUC |
| Neural network | Can learn feature interactions trees miss | Unknown — worth testing |

### Explicitly Avoided
- **Betting odds** — encodes market consensus of features already built; creates dependency on external data and doesn't represent independent signal discovery.
