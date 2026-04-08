import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score
from pathlib import Path
import json

DATA = Path(__file__).parent.parent.parent / "data" / "features.csv"
OUT  = Path(__file__).parent

META_COLS = [
    "meta_root_fight_id", "meta_f1_id", "meta_f2_id",
    "meta_winner_id", "meta_loser_id",
    "meta_fight_date", "meta_end_time", "meta_method", "meta_fight_type",
]


def load() -> tuple[np.ndarray, np.ndarray, list[str], pl.Series]:
    df = pl.read_csv(DATA)

    # Drop draws / no-contests (null winner_id)
    df = df.filter(pl.col("meta_winner_id").is_not_null())

    # Target: 1 if f1 wins, 0 if f2 wins
    y = (df["meta_winner_id"] == df["meta_f1_id"]).cast(pl.Int8).to_numpy()

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    dates = df["meta_fight_date"]

    return X, y, feature_cols, dates


def time_split(X, y, dates, test_frac=0.2):
    """Chronological train/test split — no future leakage."""
    sorted_idx = np.argsort(dates.to_numpy())
    cutoff = int(len(sorted_idx) * (1 - test_frac))
    train_idx = sorted_idx[:cutoff]
    test_idx  = sorted_idx[cutoff:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mirror_augment(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Double the training set by adding a f1/f2-swapped copy of every fight.

    The UFC dataset assigns f1/f2 labels by a convention that shifted over
    time — train has 56% f1 wins, test only 43%.  Without correction the
    model learns a spurious directional bias.

    For each fight we create a mirror where:
      - f1_* and f2_* columns are swapped
      - *_diff columns are negated  (diff = f1-f2, so flipped diff = f2-f1 = -diff)
      - symmetric context columns (weight_class_id, fight_format, etc.) are unchanged
      - the label is flipped  (f1 win → f2 win)

    After augmentation the training win rate is exactly 50% by construction.
    """
    col_idx = {col: i for i, col in enumerate(feature_cols)}

    # Build swap pairs: f1_XXX <-> f2_XXX
    swap_pairs: list[tuple[int, int]] = []
    seen_f2: set[str] = set()
    for col in feature_cols:
        if col.startswith("f1_"):
            partner = "f2_" + col[3:]
            if partner in col_idx and partner not in seen_f2:
                swap_pairs.append((col_idx[col], col_idx[partner]))
                seen_f2.add(partner)

    # Collect diff column indices (to negate)
    diff_cols: list[int] = [
        i for i, col in enumerate(feature_cols) if col.endswith("_diff")
    ]

    X_mirror = X.copy()

    # Swap f1/f2 pairs
    for i, j in swap_pairs:
        X_mirror[:, i], X_mirror[:, j] = X[:, j].copy(), X[:, i].copy()

    # Negate diffs
    X_mirror[:, diff_cols] = -X[:, diff_cols]

    y_mirror = 1 - y

    # Concatenate and shuffle so original/mirror pairs aren't adjacent
    X_aug = np.concatenate([X, X_mirror], axis=0)
    y_aug = np.concatenate([y, y_mirror], axis=0)

    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def train():
    X, y, feature_cols, dates = load()
    X_train, X_test, y_train, y_test = time_split(X, y, dates)

    print(f"Features : {X.shape[1]}")
    print(f"Train    : {len(y_train)} fights  ({y_train.mean():.3f} f1 win rate)  [pre-augment]")
    print(f"Test     : {len(y_test)} fights  ({y_test.mean():.3f} f1 win rate)")

    X_train, y_train = mirror_augment(X_train, y_train, feature_cols)
    print(f"Train    : {len(y_train)} fights  ({y_train.mean():.3f} f1 win rate)  [post-augment]")
    print()

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    train_proba = model.predict_proba(X_train)[:, 1]
    train_preds = (train_proba >= 0.5).astype(int)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_preds)
    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
    ll   = log_loss(y_test, proba)

    print(f"\n{'='*40}")
    print(f"Train accuracy: {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"Test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"Overfit gap   : {(train_acc - acc)*100:.1f}pp")
    print(f"Test AUC      : {auc:.4f}")
    print(f"Test log-loss : {ll:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"{'='*40}\n")

    # Feature importance
    importance = model.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

    print("Top 50 features by gain:")
    print(f"{'Rank':<5} {'Feature':<45} {'Importance':>10}")
    print("-" * 62)
    for i, (feat, imp) in enumerate(ranked[:50], 1):
        print(f"{i:<5} {feat:<45} {imp:>10.4f}")

    # Save importance to JSON for analysis
    importance_dict = {feat: float(imp) for feat, imp in ranked}
    with open(OUT / "feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)

    # Save model
    model.save_model(OUT / "model.ubj")
    print(f"\nModel saved to {OUT / 'model.ubj'}")
    print(f"Feature importance saved to {OUT / 'feature_importance.json'}")

    return model, ranked


if __name__ == "__main__":
    train()
