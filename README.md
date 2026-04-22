# Predicting UFC Fight Outcomes Using Machine Learning

**COMP 5630/6630 — Machine Learning, Auburn University**  
Clay Ramey · Nicholas Winfrey · Muhammad Khattak · Erin Houston

---

## Project Overview

This project formulates UFC fight outcome prediction as a supervised binary classification problem. Given pre-fight statistics for two fighters, the goal is to predict whether Fighter 1 wins the matchup.

- **Dataset:** 5,185 historical UFC fights (`data/fight_snapshots.parquet`)
- **Features:** 309 engineered predictors (biometrics, historical performance aggregates, f1-f2 difference features)
- **Models evaluated:** Logistic Regression, SVC, KNN, MLP, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost
- **Best result:** Logistic Regression — Test ROC-AUC **0.6348**, Accuracy **58.73%**

See the full report in `documents/Comp6630 UFC Final Project Report.pdf`.

---

## Repository Structure

```
.
├── 01_MAIN_ufc_fight_prediction.ipynb    # PRIMARY: all models, SHAP analysis, results
├── 02_exploration_feature_engineering.ipynb  # Exploratory: Polars feature pipeline
├── 03_exploration_elo_ratings.ipynb          # Exploratory: ELO rating system
│
├── data/
│   ├── fight_snapshots.parquet   # Raw fight data (nested prior fight histories)
│   ├── features.csv              # Pre-generated feature matrix (5185 x 311)
│   ├── sample_fight.json         # Example raw fight record (schema reference)
│   └── SCHEMA.md                 # Raw data schema documentation
│
├── cache/
│   └── grid_results.pkl          # Cached GridSearchCV results (avoids ~2hr retraining)
│
├── documents/
│   ├── Comp6630 UFC Final Project Report.pdf
│   └── Project Proposal.pdf
│
├── requirements.txt
└── README.md
```

---

## Notebooks

### `01_MAIN_ufc_fight_prediction.ipynb` — **Start here**

The primary deliverable. Contains:
1. **Feature engineering** — `process()` converts each raw fight record into 309 predictors using biometrics, win/loss history aggregates (mean/min/max), and f1-f2 difference features
2. **Chronological train/test split** — earliest 80% for training, most recent 20% for evaluation
3. **10-model comparison** with GridSearchCV + TimeSeriesSplit (5 folds) to prevent temporal leakage
4. **Feature importance analysis** — top 30 features for the 4 best models
5. **SHAP beeswarm error analysis** — why LightGBM misclassifies certain fights

### `02_exploration_feature_engineering.ipynb`

Exploratory notebook showing an alternative Polars-based feature pipeline with more granular feature groups: striking stats across 3 temporal windows with trend signals, round-level pacing (early vs. late), win/loss streaks, accumulated damage, recency-weighted stats, and mirror augmentation to fix a label bias in the dataset.

### `03_exploration_elo_ratings.ipynb`

Exploratory notebook implementing a custom ELO rating system for UFC fighters with variable K schedule, per-division ELO with cross-division transfer factors, domain ELOs (striking offense/defense, grappling, finishing), and peak ELO with momentum velocity features.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main notebook

Open `01_MAIN_ufc_fight_prediction.ipynb` in Jupyter. The notebook uses a cached result file (`cache/grid_results.pkl`) so all 10 GridSearchCV runs load instantly without retraining (~2 hours of compute).

To force a full retrain, delete `cache/grid_results.pkl` before running.

### 3. (Optional) Explore alternative approaches

Open `02_exploration_feature_engineering.ipynb` or `03_exploration_elo_ratings.ipynb` for the exploratory feature pipelines. These require `polars` in addition to the standard stack.

---

## Data

| File | Description |
|------|-------------|
| `data/fight_snapshots.parquet` | Raw source data. Each row is one fight with nested prior fight histories for both fighters. |
| `data/features.csv` | Pre-generated feature matrix produced by running `process()` from the main notebook. 5185 rows × 311 columns (309 features + `target` + `date`). |
| `data/sample_fight.json` | A single raw fight record in JSON format. Useful for understanding the nested data schema before processing. |
| `data/SCHEMA.md` | Full schema documentation for the raw parquet file. |

---

## Key Design Decisions

**No data leakage.** Only information available *before* each fight is used. Post-fight statistics and improperly aggregated career totals that encode future information are excluded.

**Chronological evaluation.** Training uses the earliest 80% of fights; testing uses the most recent 20%. Hyperparameter tuning uses `TimeSeriesSplit` so each validation fold is always *after* its training fold in time.

**Feature engineering approach.** Rather than raw career totals, we compute separate aggregate statistics for wins and losses (mean/min/max), capturing *how* a fighter performs in successful vs. unsuccessful bouts. This bifurcated approach expands the feature space to 309 columns.

---

## Results Summary

| Model | Test ROC-AUC | Test Accuracy | Fit Time (s) |
|-------|:---:|:---:|:---:|
| **Logistic Regression** | **0.6348** | **58.73%** | 29.91 |
| LightGBM | 0.6319 | 58.05% | 1883.14 |
| CatBoost | 0.6264 | 55.74% | 3629.90 |
| XGBoost | 0.6263 | 57.76% | 1349.49 |
| SVC | 0.6112 | 56.22% | 483.82 |
| Random Forest | 0.6102 | 56.32% | 51.80 |
| Gradient Boosting | 0.6019 | 56.80% | 113.21 |
| AdaBoost | 0.5991 | 51.78% | 19.78 |
| Neural Network (MLP) | 0.5746 | 56.12% | 29.63 |
| K-Nearest Neighbors | 0.5268 | 49.28% | 3.83 |
