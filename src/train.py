# src/train.py
# Phase 3 — Stacking Ensemble: XGBoost + LightGBM + Random Forest → Logistic Regression

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────────
# CONFIG — update DATA_FILE to match your CSV
# ─────────────────────────────────────────────
DATA_FILE   = "data/processed_urls.csv"
LABEL_COL   = "label"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
RANDOM_STATE = 42
N_FOLDS      = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP A — Load data
# ─────────────────────────────────────────────
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=[LABEL_COL])
df[LABEL_COL] = df[LABEL_COL].apply(lambda x: 0 if x == 0 else 1)
df[LABEL_COL] = df[LABEL_COL].astype(int)

df = df.dropna(subset=[LABEL_COL])

# Auto-detect label column if not found
if LABEL_COL not in df.columns:
    candidates = [c for c in df.columns if c.lower() in
                  ("label", "target", "class", "phishing", "is_phishing")]
    if not candidates:
        raise ValueError(f"Cannot find label column. Columns are: {list(df.columns)}")
    LABEL_COL = candidates[0]
    print(f"    Auto-detected label column: '{LABEL_COL}'")

feature_names = [c for c in df.columns if c != LABEL_COL]
X = df[feature_names].values.astype(np.float32)
y = df[LABEL_COL].values

print(f"    Dataset shape: {X.shape}")
print(f"    Phishing: {y.sum()}  |  Benign: {(y==0).sum()}")
print(f"    Features: {feature_names}")

# Save feature names (same format dashboard expects)
joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.pkl")
print(f"    Saved: {MODELS_DIR}/feature_names.pkl")


# ─────────────────────────────────────────────
# STEP B — Define base models
# ─────────────────────────────────────────────
print("\n[2/7] Defining base models...")

base_models = {
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ─────────────────────────────────────────────
# STEP C — 5-Fold cross-validation (paper metrics)
# ─────────────────────────────────────────────
print(f"\n[3/7] Running {N_FOLDS}-fold cross-validation on each base model...")

cv_results = {}

for name, clf in base_models.items():
    print(f"    {name}...", end=" ", flush=True)
    cv = cross_validate(clf, X, y, cv=skf, scoring=SCORING, n_jobs=-1)
    cv_results[name] = {
        metric: {
            "mean": float(np.mean(cv[f"test_{metric}"])),
            "std":  float(np.std(cv[f"test_{metric}"])),
        }
        for metric in SCORING
    }
    print(f"F1={cv_results[name]['f1']['mean']:.4f} "
          f"(+/-{cv_results[name]['f1']['std']:.4f})  "
          f"ROC-AUC={cv_results[name]['roc_auc']['mean']:.4f}")

with open(f"{RESULTS_DIR}/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"    Saved: {RESULTS_DIR}/cv_results.json")


# ─────────────────────────────────────────────
# STEP D — Generate out-of-fold predictions for stacking
# ─────────────────────────────────────────────
print("\n[4/7] Generating out-of-fold predictions for meta-learner...")

oof_preds = np.zeros((len(X), len(base_models)))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train        = y[train_idx]
    for model_idx, (name, clf) in enumerate(base_models.items()):
        clf_clone = type(clf)(**clf.get_params())
        clf_clone.fit(X_train, y_train)
        oof_preds[val_idx, model_idx] = clf_clone.predict_proba(X_val)[:, 1]
    print(f"    Fold {fold_idx + 1}/{N_FOLDS} done")

print("    Out-of-fold predictions complete.")


# ─────────────────────────────────────────────
# STEP E — Train meta-learner on OOF predictions
# ─────────────────────────────────────────────
print("\n[5/7] Training stacking meta-learner (Logistic Regression)...")

meta_learner = LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)
meta_learner.fit(oof_preds, y)

meta_weights = dict(zip(base_models.keys(), meta_learner.coef_[0]))
print(f"    Meta-learner weights: {meta_weights}")


# ─────────────────────────────────────────────
# STEP F — Train final base models on full data + calibrate
# ─────────────────────────────────────────────
print("\n[6/7] Training final base models on full dataset and calibrating...")

trained_base_models = {}

for name, clf in base_models.items():
    print(f"    Training {name}...", end=" ", flush=True)
    clf.fit(X, y)
    calibrated = CalibratedClassifierCV(clf, cv=3, method="isotonic")
    calibrated.fit(X, y)
    trained_base_models[name] = calibrated
    print("done")

# Save individual base models
joblib.dump(trained_base_models, f"{MODELS_DIR}/base_models.pkl")
print(f"    Saved: {MODELS_DIR}/base_models.pkl")

# Save meta-learner
joblib.dump(meta_learner, f"{MODELS_DIR}/meta_learner.pkl")
print(f"    Saved: {MODELS_DIR}/meta_learner.pkl")


# ─────────────────────────────────────────────
# STEP G — Evaluate stacking ensemble + save results
# ─────────────────────────────────────────────
print("\n[7/7] Evaluating final stacking ensemble...")

# Get stacked predictions on full data (for reporting)
stack_input = np.column_stack([
    model.predict_proba(X)[:, 1]
    for model in trained_base_models.values()
])
final_proba = meta_learner.predict_proba(stack_input)[:, 1]
final_pred  = (final_proba >= 0.5).astype(int)

ensemble_metrics = {
    "accuracy":  float(accuracy_score(y, final_pred)),
    "precision": float(precision_score(y, final_pred)),
    "recall":    float(recall_score(y, final_pred)),
    "f1":        float(f1_score(y, final_pred)),
    "roc_auc":   float(roc_auc_score(y, final_proba)),
}

print("\n    === STACKING ENSEMBLE RESULTS ===")
for k, v in ensemble_metrics.items():
    print(f"    {k:12s}: {v:.4f}")

# Save stacking model as best_model replacement
stacking_bundle = {
    "base_models":   trained_base_models,
    "meta_learner":  meta_learner,
    "feature_names": feature_names,
    "model_type":    "stacking_ensemble",
}
joblib.dump(stacking_bundle, f"{MODELS_DIR}/stacking_model.pkl")
print(f"\n    Saved: {MODELS_DIR}/stacking_model.pkl")

# Save full results report
full_report = {
    "cross_validation": cv_results,
    "ensemble_final":   ensemble_metrics,
    "meta_weights":     meta_weights,
    "feature_names":    feature_names,
    "n_samples":        int(len(X)),
    "n_features":       int(X.shape[1]),
    "n_folds":          N_FOLDS,
}
with open(f"{RESULTS_DIR}/training_report.json", "w") as f:
    json.dump(full_report, f, indent=2)
print(f"    Saved: {RESULTS_DIR}/training_report.json")

print("\n Phase 3 complete.")
print(f"    Models saved to: {MODELS_DIR}/")
print(f"    Results saved to: {RESULTS_DIR}/")