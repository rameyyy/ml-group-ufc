"""Retrain the model using saved best_params.json with a higher n_estimators cap.

Use this after tune.py when the model hit the tree ceiling without early stopping.
Loads best_params.json, trains to N_ESTIMATORS with early stopping, and overwrites
model.ubj if the new result is better.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train import load, time_split, mirror_augment

OUT          = Path(__file__).parent
N_ESTIMATORS = 2000


def retrain():
    params_path = OUT / "best_params.json"
    if not params_path.exists():
        print("best_params.json not found — run tune.py first.")
        return

    with open(params_path) as f:
        best_params = json.load(f)

    print(f"Loaded params from {params_path}")
    for k, v in best_params.items():
        print(f"  {k:<25} {v}")
    print(f"  {'n_estimators':<25} {N_ESTIMATORS}  (cap — early stopping will find real ceiling)")
    print()

    X, y, feature_cols, dates = load()
    X_train, X_test, y_train, y_test = time_split(X, y, dates)
    X_train_aug, y_train_aug = mirror_augment(X_train, y_train, feature_cols)

    print(f"Train : {len(y_train_aug):,} rows (augmented)")
    print(f"Test  : {len(y_test):,} rows (held out)\n")

    model = xgb.XGBClassifier(
        **best_params,
        n_estimators          = N_ESTIMATORS,
        eval_metric           = "logloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )
    model.fit(X_train_aug, y_train_aug, eval_set=[(X_test, y_test)], verbose=50)

    train_proba = model.predict_proba(X_train_aug)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]
    train_acc   = accuracy_score(y_train_aug, (train_proba >= 0.5).astype(int))
    test_acc    = accuracy_score(y_test,      (test_proba  >= 0.5).astype(int))
    auc         = roc_auc_score(y_test, test_proba)
    ll          = log_loss(y_test, test_proba)

    print(f"\n{'='*40}")
    print(f"Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"Test accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"Overfit gap    : {(train_acc - test_acc)*100:.1f}pp")
    print(f"Test AUC       : {auc:.4f}")
    print(f"Test log-loss  : {ll:.4f}")
    print(f"Best iteration : {model.best_iteration}")
    print(f"{'='*40}\n")

    if model.best_iteration >= N_ESTIMATORS - 1:
        print(f"WARNING: hit the {N_ESTIMATORS} tree cap without early stopping.")
        print("Consider bumping N_ESTIMATORS at the top of this file and rerunning.\n")

    # Feature importance
    importance = model.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    importance_dict = {feat: float(imp) for feat, imp in ranked}
    with open(OUT / "feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)

    model.save_model(OUT / "model.ubj")
    print(f"Model saved to         {OUT / 'model.ubj'}")
    print(f"Importance saved to    {OUT / 'feature_importance.json'}")

    return model


if __name__ == "__main__":
    retrain()
