# src/threshold.py
# Phase 4 — Data-driven adaptive threshold with uncertainty zone

import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score

MODELS_DIR  = "models"
RESULTS_DIR = "results"
UNCERTAINTY_BAND = 0.08   # +/- this around threshold = "uncertain"
HIGH_RISK_SHIFT  = 0.05   # lower threshold by this when URL has IP or suspicious TLD


def compute_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the threshold that maximises F1 score on the validation set.
    This replaces the hardcoded 0.75.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls)
    )

    # thresholds has one fewer element than precisions/recalls
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])

    return best_threshold, best_f1


def get_adaptive_threshold(base_threshold: float,
                            has_ip: int,
                            suspicious_tld: int) -> float:
    """
    Shift the base threshold downward if high-risk URL signals are present.
    Lower threshold = more aggressive detection.

    Rule:
      - IP address in URL   → lower threshold by HIGH_RISK_SHIFT
      - Suspicious TLD      → lower threshold by HIGH_RISK_SHIFT
      - Both present        → lower by HIGH_RISK_SHIFT (not double-counted)
      - Neither             → use base threshold as-is
    """
    if has_ip or suspicious_tld:
        return max(0.3, base_threshold - HIGH_RISK_SHIFT)
    return base_threshold


def classify_with_uncertainty(prob: float,
                               threshold: float) -> dict:
    """
    Given a probability and threshold, return:
      - decision     : 'phishing' | 'safe' | 'uncertain'
      - in_uncertainty_zone : bool
      - lower_bound  : threshold - UNCERTAINTY_BAND
      - upper_bound  : threshold + UNCERTAINTY_BAND
    """
    lower = threshold - UNCERTAINTY_BAND
    upper = threshold + UNCERTAINTY_BAND

    in_zone = lower <= prob <= upper

    if in_zone:
        decision = "uncertain"
    elif prob > upper:
        decision = "phishing"
    else:
        decision = "safe"

    return {
        "decision":             decision,
        "in_uncertainty_zone":  in_zone,
        "threshold":            round(threshold, 4),
        "lower_bound":          round(lower, 4),
        "upper_bound":          round(upper, 4),
    }


def run_threshold_computation():
    """
    Full pipeline:
    1. Load stacking model
    2. Load data
    3. Generate hold-out validation predictions
    4. Compute optimal threshold
    5. Save threshold.pkl and threshold_report.json
    """

    # ── Load model bundle ──────────────────────────────────
    print("\n[1/4] Loading stacking model...")
    bundle       = joblib.load(f"{MODELS_DIR}/stacking_model.pkl")
    base_models  = bundle["base_models"]
    meta_learner = bundle["meta_learner"]
    feature_names = bundle["feature_names"]

    # ── Load data ──────────────────────────────────────────
    print("[2/4] Loading dataset for threshold computation...")

    # Find the processed CSV automatically
    data_file = "data/processed_urls.csv"
    print(f"    Using: {data_file}")

    df = pd.read_csv(data_file)
    # ---------------- FIX LABELS ----------------
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)
    df["label"] = df["label"].astype(int)
    # -------------------------------------------

    # Auto-detect label column
    label_col = next(
        (c for c in df.columns if c.lower() in
         ("label", "target", "class", "phishing", "is_phishing")),
        None
    )
    if label_col is None:
        raise ValueError(f"Cannot find label column in: {list(df.columns)}")

    X = df[feature_names].values.astype("float32")
    y = df[label_col].values

    # Hold-out split — 20% for threshold computation only
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    Validation set: {len(X_val)} samples")

    # ── Generate stacking probabilities on validation set ──
    print("[3/4] Generating validation predictions...")

    stack_input = np.column_stack([
        m.predict_proba(X_val)[:, 1]
        for m in base_models.values()
    ])
    val_proba = meta_learner.predict_proba(stack_input)[:, 1]

    # ── Compute optimal threshold ──────────────────────────
    print("[4/4] Computing optimal threshold...")

    best_threshold, best_f1 = compute_threshold(y_val, val_proba)

    # Also compute F1 at default 0.75 for comparison
    old_preds = (val_proba >= 0.75).astype(int)
    old_f1    = float(f1_score(y_val, old_preds))

    new_preds = (val_proba >= best_threshold).astype(int)
    new_f1    = float(f1_score(y_val, new_preds))

    print(f"\n    Hardcoded threshold (0.75) F1 : {old_f1:.4f}")
    print(f"    Optimal  threshold ({best_threshold:.4f}) F1 : {new_f1:.4f}")
    print(f"    Improvement: +{(new_f1 - old_f1):.4f}")
    print(f"\n    Uncertainty zone: "
          f"[{best_threshold - UNCERTAINTY_BAND:.4f}, "
          f"{best_threshold + UNCERTAINTY_BAND:.4f}]")
    print(f"    Adaptive shift for high-risk URLs: -{HIGH_RISK_SHIFT}")

    # ── Save threshold ────────────────────────────────────
    threshold_data = {
        "base_threshold":     best_threshold,
        "uncertainty_band":   UNCERTAINTY_BAND,
        "high_risk_shift":    HIGH_RISK_SHIFT,
        "lower_bound":        best_threshold - UNCERTAINTY_BAND,
        "upper_bound":        best_threshold + UNCERTAINTY_BAND,
        "f1_at_threshold":    new_f1,
        "f1_at_0_75":         old_f1,
        "improvement":        new_f1 - old_f1,
        "val_set_size":       int(len(X_val)),
    }

    joblib.dump(threshold_data, f"{MODELS_DIR}/threshold.pkl")
    print(f"\n    Saved: {MODELS_DIR}/threshold.pkl")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/threshold_report.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"    Saved: {RESULTS_DIR}/threshold_report.json")

    print("\n Phase 4 complete.")
    return threshold_data


if __name__ == "__main__":
    run_threshold_computation()