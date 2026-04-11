# src/pipeline.py
# Phase 5 — PhishGuardPipeline: unified URL + behavioral fusion

import numpy as np
import pandas as pd
import joblib

from src.feature_extractor import extract_features
from src.behavior_model import compute_behavior_score
from src.threshold import get_adaptive_threshold, classify_with_uncertainty

# Fusion weights — documented in paper
URL_WEIGHT      = 0.65
BEHAVIOR_WEIGHT = 0.35


class PhishGuardPipeline:
    """
    Unified phishing detection pipeline.

    Combines:
      - Structural URL features (21 signals) → stacking ensemble
      - Behavioral session signals (7 inputs) → weighted scorer
      - Adaptive threshold with uncertainty zone

    Usage:
        pipeline = PhishGuardPipeline.load()
        result   = pipeline.predict(url, behavior_data={...})
    """

    def __init__(self, base_models, meta_learner, feature_names, threshold_data):
        self.base_models    = base_models
        self.meta_learner   = meta_learner
        self.feature_names  = feature_names
        self.threshold_data = threshold_data

    @classmethod
    def load(cls,
             model_path="models/stacking_model.pkl",
             threshold_path="models/threshold.pkl"):
        bundle         = joblib.load(model_path)
        threshold_data = joblib.load(threshold_path)
        return cls(
            base_models    = bundle["base_models"],
            meta_learner   = bundle["meta_learner"],
            feature_names  = bundle["feature_names"],
            threshold_data = threshold_data,
        )

    def _url_score(self, url: str) -> tuple[float, dict]:
        """Extract features and return stacking model probability + raw features."""
        raw = extract_features(url)
        df  = pd.DataFrame([raw])[self.feature_names]

        stack_input = np.column_stack([
            m.predict_proba(df)[:, 1]
            for m in self.base_models.values()
        ])
        prob = float(self.meta_learner.predict_proba(stack_input)[0][1])
        return prob, raw

    def predict(self, url: str, behavior_data: dict = None) -> dict:
        """
        Run full pipeline on a URL with optional behavioral data.

        behavior_data keys (all optional, defaults to low-risk values):
            session_duration, time_to_submit, num_pages,
            scroll_depth, mouse_variance, back_button, tab_switches

        Returns:
            url_score        : float [0,1] — raw model probability
            behavior_score   : float [0,1] — behavioral risk (0 if no data)
            final_score      : float [0,1] — fused score
            decision         : 'phishing' | 'uncertain' | 'safe'
            in_uncertainty_zone : bool
            threshold_used   : float
            url_features     : dict of 21 features
            behavior_result  : dict of behavioral components
            fusion_weights   : dict showing URL/behavior weights applied
        """

        # ── URL score ────────────────────────────────────
        url_score, raw_features = self._url_score(url)

        # ── Behavioral score ─────────────────────────────
        if behavior_data:
            beh = compute_behavior_score(
                session_duration = behavior_data.get("session_duration", 60),
                time_to_submit   = behavior_data.get("time_to_submit", 30),
                num_pages        = behavior_data.get("num_pages", 5),
                scroll_depth     = behavior_data.get("scroll_depth", 50),
                mouse_variance   = behavior_data.get("mouse_variance", 50),
                back_button      = behavior_data.get("back_button", 0),
                tab_switches     = behavior_data.get("tab_switches", 0),
            )
            behavior_score    = beh["behavior_score"]
            behavior_result   = beh
            applied_url_w     = URL_WEIGHT
            applied_beh_w     = BEHAVIOR_WEIGHT
        else:
            # URL-only mode — full weight on URL score
            behavior_score  = 0.0
            behavior_result = {}
            applied_url_w   = 1.0
            applied_beh_w   = 0.0

        # ── Fusion ───────────────────────────────────────
        final_score = (applied_url_w * url_score) + (applied_beh_w * behavior_score)
        final_score = float(min(max(final_score, 0.0), 1.0))

        # ── Adaptive threshold ───────────────────────────
        adaptive_thresh = get_adaptive_threshold(
            base_threshold = self.threshold_data["base_threshold"],
            has_ip         = raw_features["has_ip"],
            suspicious_tld = raw_features["suspicious_tld"],
        )

        # ── Decision ─────────────────────────────────────
        result = classify_with_uncertainty(final_score, adaptive_thresh)

        return {
            "url_score":            round(url_score, 4),
            "behavior_score":       round(behavior_score, 4),
            "final_score":          round(final_score, 4),
            "decision":             result["decision"],
            "in_uncertainty_zone":  result["in_uncertainty_zone"],
            "threshold_used":       round(adaptive_thresh, 4),
            "lower_bound":          result["lower_bound"],
            "upper_bound":          result["upper_bound"],
            "url_features":         raw_features,
            "behavior_result":      behavior_result,
            "fusion_weights":       {
                "url":      applied_url_w,
                "behavior": applied_beh_w,
            },
        }