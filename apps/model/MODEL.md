# Model Development

## Baseline Results

| Metric | Value |
|---|---|
| Train accuracy | 84.2% |
| Test accuracy | 72.4% |
| Overfit gap | 11.8pp |
| Test AUC | 0.818 |
| Test log-loss | 0.539 |
| Best iteration | 197 / 400 |
| Features | 125 |
| Train rows | 4,086 |
| Test rows | 1,022 |

**Split:** Chronological 80/20 — oldest 80% of fights train, most recent 20% test. No shuffling, zero future leakage.

**Note on class shift:** Train has 56% f1 wins, test has only 43%. This is a data artifact — how f1/f2 get assigned likely changed over time (ordering convention, etc.). It makes the test set harder by design and slightly inflates the apparent overfit gap. Worth investigating before drawing strong conclusions.

---

## Diagnosis

The **11.8pp train/test gap** indicates real overfitting. Not catastrophic — early stopping is helping — but the model has memorized training patterns that don't fully generalize. The 125-feature / 4k-row ratio is tight, and some features likely carry noise at that scale.

---

## Potential Features (Brainstormed)

### Style Fingerprinting / Matchup
| Feature | Idea |
|---|---|
| **Wrestler vs Striker score** | `(td_avg * ctrl_time) vs (slpm * str_acc)` → continuous grappler-to-striker index per fighter, diff for matchup |
| **Reach utilization** | `distance_landed / sig_str_landed` — does the fighter actually fight at range? Interact with reach_diff |
| **Clinch-to-takedown pipeline** | `td_landed / clinch_attempts` — clinch-to-takedown conversion rate |
| **Ground control quality** | `ground_str_landed / ctrl_time_s` — separates wrestlers-who-hold from wrestlers-who-punish |

### Wear and Tear / Damage
| Feature | Idea |
|---|---|
| **Career damage index** | Cumulative `opp_sig_str_landed` across all prior fights, recency-weighted — chin erosion proxy |
| **KO chin trend** | `opp_kd` rate in last 3 vs career — is the chin holding up recently? |
| **Head shot concentration** | `opp_head_landed / opp_sig_str_landed` — absorbing clean head shots vs leg kicks; different long-term profiles |

### Strength of Schedule
| Feature | Idea |
|---|---|
| **Avg opponent ELO** | Mean ELO of all prior opponents at fight time — same win rate, very different if they beat cans vs contenders |
| **Performance vs elite** | slpm / str_acc specifically when facing opponents with ELO > 1600 — do they level up or get exposed? |

### Grappling Depth
| Feature | Idea |
|---|---|
| **Sub threat from bottom** | `sub_att` filtered to rounds where `opp_ctrl_time_s > 0` — threatens from guard vs just survives |
| **Takedown-to-sub conversion** | `sub_att / td_landed` — once on the mat, how dangerous? |

### Mental / Momentum
| Feature | Idea |
|---|---|
| **Finish rate trend** | `finish_rate_last_3 / finish_rate_career` — going up = peaking, down = getting into wars |
| **Big fight performance** | slpm / str_acc specifically in 5-round fights — rises to occasion or gasses? |
| **Return from layoff** | Win rate in fights after layoff > 1yr, per fighter — fighter-specific rust signal |

---

## Priority Roadmap

### 1. Feature Engineering — Highest ROI, do first

At 72% accuracy with 125 features, the model is near the ceiling of what raw historical stats can give. High-signal features that are missing:

| Feature | Why it matters |
|---|---|
| **Betting odds** | Single strongest predictor in sports ML. Encodes market consensus on everything we've engineered plus intangibles. Opening vs. closing line movement is a separate signal. |
| **ELO rating** ✓ | Implemented with variable K schedule tuned via bucket-MAE calibration. Schedule: 0–4 fights K=66, 5–14 K=68, 15–19 K=63, 20+ K=54. Cal loss 0.0049 vs 0.0366 for fixed K=32 (7.5x improvement). Stat blending (alpha>0) consistently hurt — UFC upsets are frequent enough that binary result carries more signal. Top 10: Makhachev, Jones, Khabib, Holloway, Dvalishvili, Usman, Volkanovski, Topuria, Chimaev, DC. |
| **Win/loss streak** | Current momentum. Equal records can hide very different recent trajectories. |
| **Short-turnaround flag** | `years_since_last_fight` misses the non-linear risk at the short end — a 45-day turnaround is very different from 6 months. |
| **Style matchup** | Grappler vs. striker, etc. Derivable from existing stats (high td_avg + low slpm ≈ wrestler). The model can learn this indirectly but explicit interaction features help tree models. |

### 2. Tighten Regularization — Low effort, meaningful gain

The overfit gap says there's room to regularize before adding tuning infrastructure. Suggested parameter adjustments:

```python
max_depth=3,           # was 4 — biggest single lever against overfit in XGBoost
min_child_weight=10,   # was 5 — require more data per leaf
gamma=2.0,             # was 1.0 — harder split threshold
colsample_bytree=0.6,  # was 0.8 — less feature exposure per tree
subsample=0.7,         # was 0.8
```

Expected impact: close 3-5pp of the gap, likely landing closer to 74-75% test accuracy.

### 3. Bayesian Hyperparameter Tuning — After regularizing

**Optuna** is the right tool — faster and smarter than grid search. Critical constraint: must use **time-series cross-validation**, not random k-fold, or CV scores will be leaky.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# fold 1: train fights 1-800,  val 801-1000
# fold 2: train fights 1-1600, val 1601-2000
# fold 3: train fights 1-2400, val 2401-3000
# ...
```

Search space: `max_depth`, `learning_rate`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`. Run 50-100 trials, optimize on val AUC.

Expected gain: **1-3% accuracy**, but more importantly better-calibrated probabilities.

### 4. Ensemble — Do last, smallest marginal gain

Once base models are well-tuned, ensemble for the final squeeze:

- **Averaged probabilities** across XGBoost + LightGBM + CatBoost — the three often disagree in complementary ways, making their average more robust than any single model
- **Stacking** (logistic meta-learner on out-of-fold predictions) can squeeze a bit more but adds pipeline complexity

Realistic gain over a well-tuned single model: **0.5-1.5%**.

---

## Suggested Execution Order

```
Phase 1 — Features
  - ELO rating per fighter at fight time
  - Win/loss streak (current streak length + direction)
  - Betting odds if obtainable
  - Short-turnaround flag (days_since_last_fight < 60)
  → Regenerate features.csv, retrain baseline

Phase 2 — Regularization
  - Tighten max_depth, min_child_weight, gamma manually
  - Recheck train/test gap; target < 8pp

Phase 3 — Tuning
  - Implement Optuna + TimeSeriesSplit CV
  - 50-100 trials on full search space
  - Evaluate on held-out test set (not CV) for final numbers

Phase 4 — Ensemble (if AUC > 0.83 on single model)
  - Train LightGBM and CatBoost with equivalent hyperparams
  - Blend probabilities (simple average or learned weights)
  - Evaluate ensemble vs. best single model
```

---

## Top 50 Features (Baseline Run)

| Rank | Feature | Importance |
|---|---|---|
| 1 | win_rate_diff | 0.0284 |
| 2 | f1_sub_losses_at_weight | 0.0173 |
| 3 | rd_dom_rate_diff | 0.0147 |
| 4 | f2_sub_losses_at_weight | 0.0146 |
| 5 | last_rd_str_diff_diff | 0.0142 |
| 6 | f2_fights_last_3yrs | 0.0134 |
| 7 | f1_5rd_losses | 0.0131 |
| 8 | age_diff | 0.0127 |
| 9 | f2_fight_count | 0.0119 |
| 10 | f1_fight_count | 0.0108 |
| 11 | f2_3rd_wins | 0.0106 |
| 12 | late_sub_per_round_diff | 0.0105 |
| 13 | f1_fights_at_weight | 0.0102 |
| 14 | f2_avg_fights_per_year | 0.0101 |
| 15 | f1_fights_last_3yrs | 0.0098 |
| 16 | f1_5rd_fights | 0.0094 |
| 17 | td_avg_diff | 0.0093 |
| 18 | reversal_rate_diff | 0.0092 |
| 19 | f2_sub_win_rate | 0.0091 |
| 20 | f1_ko_wins_at_weight | 0.0091 |
| 21 | post_kd_response_diff | 0.0090 |
| 22 | f1_sub_win_rate | 0.0090 |
| 23 | f1_sub_wins_at_weight | 0.0089 |
| 24 | f1_avg_fights_per_year | 0.0089 |
| 25 | last_3_sub_avg_diff | 0.0089 |
| 26 | f1_age | 0.0087 |
| 27 | f2_years_since_last_fight | 0.0087 |
| 28 | last_fight_slpm_diff | 0.0086 |
| 29 | f1_years_since_last_fight | 0.0086 |
| 30 | sub_avg_diff | 0.0086 |
| 31 | f2_last_3_avg_end_time_s | 0.0086 |
| 32 | td_def_trend_diff | 0.0086 |
| 33 | f1_ko_losses_at_weight | 0.0085 |
| 34 | last_fight_kd_rate_diff | 0.0085 |
| 35 | f1_3rd_losses | 0.0085 |
| 36 | str_def_diff | 0.0085 |
| 37 | f2_5rd_fights | 0.0085 |
| 38 | last_3_sapm_diff | 0.0085 |
| 39 | sapm_diff | 0.0084 |
| 40 | reach_diff | 0.0084 |
| 41 | last_3_r1_sig_per_fight_diff | 0.0083 |
| 42 | f1_stance_id | 0.0082 |
| 43 | td_acc_trend_diff | 0.0081 |
| 44 | last_fight_str_def_diff | 0.0081 |
| 45 | td_avg_trend_diff | 0.0081 |
| 46 | f1_sub_loss_rate | 0.0081 |
| 47 | f2_age | 0.0080 |
| 48 | weight_class_id | 0.0080 |
| 49 | last_fight_sub_avg_diff | 0.0080 |
| 50 | last_3_str_acc_degradation_diff | 0.0079 |
