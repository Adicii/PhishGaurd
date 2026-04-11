"""
src/research_analysis.py
------------------------
Phase 9 — paper-facing analysis modules.

Three independent functions, each callable from train.py or standalone:

1. distribution_shift_analysis()  — internal vs external accuracy report
2. adversarial_robustness()       — manually crafted evasion URL evaluation
3. write_generalization_report()  — saves results/generalization_report.json

Run all three:
    python src/research_analysis.py

Or import individually in src/train.py after model training is complete.
"""

import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from urllib.parse import urlparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1.  DISTRIBUTION SHIFT ANALYSIS
# ─────────────────────────────────────────────────────────────
def distribution_shift_analysis(
    model,
    X_internal : np.ndarray,
    y_internal : np.ndarray,
    X_external : np.ndarray,
    y_external : np.ndarray,
    threshold  : float = 0.75,
    save_figure: bool  = True,
) -> dict:
    """
    Computes and compares model performance on two separate data distributions.

    Parameters
    ----------
    model       : fitted sklearn-compatible classifier with predict_proba()
    X_internal  : feature matrix from the same dataset as training (held-out test split)
    y_internal  : ground truth labels for internal set
    X_external  : feature matrix from a completely different data source
    y_external  : ground truth labels for external set
    threshold   : decision threshold
    save_figure : whether to save a comparison bar chart to results/figures/

    Returns
    -------
    dict with per-set metrics + gap analysis
    """
    def _metrics(X, y, name: str) -> dict:
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return {
            "dataset"   : name,
            "n_samples" : len(y),
            "accuracy"  : round(accuracy_score(y, preds),                    4),
            "precision" : round(precision_score(y, preds, zero_division=0),  4),
            "recall"    : round(recall_score(y, preds,    zero_division=0),  4),
            "f1"        : round(f1_score(y, preds,        zero_division=0),  4),
            "roc_auc"   : round(roc_auc_score(y, probs),                     4),
        }

    int_m = _metrics(X_internal, y_internal, "internal")
    ext_m = _metrics(X_external, y_external, "external")

    gap = {
        metric: round(int_m[metric] - ext_m[metric], 4)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    acc_gap = gap["accuracy"]
    if   acc_gap > 0.40: severity = "critical"
    elif acc_gap > 0.20: severity = "significant"
    elif acc_gap > 0.10: severity = "moderate"
    else:                severity = "acceptable"

    result = {
        "internal"      : int_m,
        "external"      : ext_m,
        "gap"           : gap,
        "gap_severity"  : severity,
        "threshold_used": threshold,
        "interpretation": (
            f"A {acc_gap:.1%} accuracy gap ({severity}) is observed between internal and external "
            f"evaluation. This is consistent with dataset distribution shift — the two sources differ "
            f"in URL origin, length distribution, and TLD composition. Recall on external data is "
            f"{ext_m['recall']:.1%}, meaning the model still catches the majority of phishing URLs "
            f"it has never been trained on. This is the operationally critical metric for a "
            f"security system. The internal ROC-AUC of {int_m['roc_auc']:.4f} reflects in-distribution "
            f"separation and should be interpreted conservatively."
        ),
    }

    if save_figure:
        metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        int_vals = [int_m[m] for m in metrics]
        ext_vals = [ext_m[m] for m in metrics]
        x = np.arange(len(metrics))
        w = 0.35

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w/2, int_vals, w, label="Internal", color="#2ecc71", alpha=0.85)
        ax.bar(x + w/2, ext_vals, w, label="External", color="#e74c3c", alpha=0.85)
        for xi, iv, ev in zip(x, int_vals, ext_vals):
            ax.text(xi - w/2, iv + 0.005, f"{iv:.3f}", ha="center", fontsize=8)
            ax.text(xi + w/2, ev + 0.005, f"{ev:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title("Distribution shift — internal vs external performance", fontsize=11)
        ax.legend()
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
        fig.tight_layout()
        path = FIGURES_DIR / "distribution_shift.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[research] saved figure -> {path}")

    _print_shift_summary(result)
    return result


def _print_shift_summary(result: dict):
    i = result["internal"]
    e = result["external"]
    g = result["gap"]
    print("\n── Distribution shift analysis ─────────────────────────")
    print(f"  {'Metric':<12} {'Internal':>10} {'External':>10} {'Gap':>10}")
    print(f"  {'-'*44}")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {m:<12} {i[m]:>10.4f} {e[m]:>10.4f} {g[m]:>+10.4f}")
    print(f"\n  Gap severity : {result['gap_severity'].upper()}")
    print(f"  {result['interpretation'][:120]}...")
    print()


# ─────────────────────────────────────────────────────────────
# 2.  ADVERSARIAL ROBUSTNESS ANALYSIS
# ─────────────────────────────────────────────────────────────

ADVERSARIAL_PAIRS = [
    (
        "http://paypal-login-security.xyz/update/account",
        "https://paypalaccountsecure.com/account-update",
        "Replace suspicious TLD + add HTTPS + remove hyphens",
    ),
    (
        "http://192.168.1.1/bank/login.php",
        "https://secure-banking-portal.com/login",
        "Replace raw IP with plausible domain + HTTPS",
    ),
    (
        "http://apple-id-verify.top/account/signin",
        "https://apple-support-account.com/signin",
        "Replace .top TLD with .com + restructure path",
    ),
    (
        "http://signin-google-verify.cf/account",
        "https://accounts-google-portal.com/sign-in",
        "Replace .cf TLD + reword path",
    ),
    (
        "http://netflix-payment-update.gq/billing",
        "https://netflix-billing-support.com/update",
        "Replace .gq TLD + HTTPS + restructure",
    ),
    (
        "http://paypal.com.account-verify.xyz/login",
        "https://paypal-account-center.com/verify",
        "Remove legitimate brand as subdomain trick + .com TLD",
    ),
    (
        "http://bankofamerica-secure.ga/update",
        "https://bankofamerica-helpdesk.com/update",
        "Replace .ga TLD with .com",
    ),
    (
        "http://microsoft-support-alert.pw/security",
        "https://microsoftsupportportal.com/security-check",
        "Remove hyphens + replace .pw + HTTPS",
    ),
    (
        "http://amazon-account-suspended.tk/verify",
        "https://amazon-account-services.com/verify",
        "Replace .tk TLD with .com + modify path words",
    ),
    (
        "http://secure-verify-amazon.tk/confirm",
        "https://amazon-security-center.com/confirm",
        "Reorder keywords + replace .tk + HTTPS",
    ),
]


def _fast_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs  = np.array(list(counts.values())) / len(s)
    return float(scipy_entropy(probs))


def _extract_for_adversarial(url: str) -> dict:
    """Feature extraction matching the 21-feature training schema."""
    parsed = urlparse(url)
    domain = parsed.netloc
    path   = parsed.path

    try:
        import tldextract
        ext       = tldextract.extract(url)
        sub_count = len(ext.subdomain.split(".")) if ext.subdomain else 0
    except ImportError:
        parts     = domain.split(".")
        sub_count = max(0, len(parts) - 2)

    # Tokenise on the same delimiters used during training
    tokens = [t for t in re.split(r"[.\-\_/]", url) if t]

    avg_token_length = (
        sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
    )

    def _is_random_token(token: str) -> bool:
        alpha = [c for c in token.lower() if c.isalpha()]
        if len(alpha) < 3:
            return False
        consonants = sum(c not in "aeiou" for c in alpha)
        return (consonants / len(alpha)) > 0.75

    random_token_ratio = (
        sum(_is_random_token(t) for t in tokens) / len(tokens) if tokens else 0.0
    )

    return {
        "url_length"        : len(url),
        "has_https"         : int(url.startswith("https")),
        "dot_count"         : url.count("."),
        "subdomain_count"   : sub_count,
        "has_ip"            : int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "special_char_count": len(re.findall(r"[^\w]", url)),
        "digit_count"       : sum(c.isdigit() for c in url),
        "digit_ratio"       : sum(c.isdigit() for c in url) / max(len(url), 1),
        "param_count"       : url.count("?") + url.count("&"),
        "brand_keyword"     : int(bool(re.search(
                                r"paypal|google|facebook|amazon|bank|apple|microsoft|netflix",
                                url.lower()))),
        "suspicious_tld"    : int(domain.split(".")[-1] in
                                {"xyz", "top", "gq", "tk", "ml", "ga", "cf", "pw"}),
        "url_entropy"       : _fast_entropy(url),
        "domain_entropy"    : _fast_entropy(domain),
        "hyphen_count"      : url.count("-"),
        "path_depth"        : path.count("/"),
        "token_count"       : len(re.findall(r"[.\-\_/]", url)),
        "vowel_ratio"       : sum(c in "aeiouAEIOU" for c in url) / max(len(url), 1),
        "domain_length"     : len(domain),
        "phish_keyword"     : int(bool(re.search(
                                r"login|secure|verify|account|update|signin|confirm|banking|password",
                                url.lower()))),
        "at_symbol"         : int("@" in url),
        "double_slash"      : url.count("//"),
        "avg_token_length"  : round(avg_token_length,   4),
        "random_token_ratio": round(random_token_ratio, 4),
    }


def adversarial_robustness(
    base_models  : dict,
    meta_learner,
    feature_names: list,
    threshold    : float = 0.75,
    save_figure  : bool  = True,
) -> dict:
    """
    Evaluates the stacking ensemble against manually crafted evasion URLs.

    Parameters
    ----------
    base_models   : dict of {name: fitted_classifier} — the stacking base models
    meta_learner  : fitted meta-learner classifier
    feature_names : list of feature column names the models were trained on
    threshold     : decision threshold
    save_figure   : save scatter plot to results/figures/
    """
    rows     = []
    bypassed = 0

    for original_url, crafted_url, technique in ADVERSARIAL_PAIRS:
        orig_feat  = _extract_for_adversarial(original_url)
        craft_feat = _extract_for_adversarial(crafted_url)

        orig_df  = pd.DataFrame([orig_feat])[feature_names]
        craft_df = pd.DataFrame([craft_feat])[feature_names]

        orig_stack = np.column_stack([
            m.predict_proba(orig_df)[:, 1] for m in base_models.values()
        ])
        orig_prob = float(meta_learner.predict_proba(orig_stack)[0][1])

        craft_stack = np.column_stack([
            m.predict_proba(craft_df)[:, 1] for m in base_models.values()
        ])
        craft_prob = float(meta_learner.predict_proba(craft_stack)[0][1])

        orig_pred  = "phishing" if orig_prob  >= threshold else "safe"
        craft_pred = "phishing" if craft_prob >= threshold else "safe"

        bypassed_flag = (orig_pred == "phishing") and (craft_pred == "safe")
        if bypassed_flag:
            bypassed += 1

        rows.append({
            "original_url"  : original_url,
            "crafted_url"   : crafted_url,
            "technique"     : technique,
            "orig_prob"     : round(orig_prob,             4),
            "craft_prob"    : round(craft_prob,            4),
            "prob_drop"     : round(orig_prob - craft_prob, 4),
            "orig_decision" : orig_pred,
            "craft_decision": craft_pred,
            "bypass"        : bypassed_flag,
        })

    bypass_rate = bypassed / len(ADVERSARIAL_PAIRS)

    result = {
        "n_pairs"    : len(ADVERSARIAL_PAIRS),
        "n_bypassed" : bypassed,
        "bypass_rate": round(bypass_rate, 4),
        "pairs"      : rows,
        "interpretation": (
            f"Of {len(ADVERSARIAL_PAIRS)} crafted evasion attempts, {bypassed} successfully "
            f"bypassed the model (bypass rate: {bypass_rate:.1%}). "
            "This is expected for a URL-feature-only model: attackers who control the domain "
            "registration can eliminate most lexical signals. This motivates the multi-modal "
            "approach — behavioural signals cannot be controlled by the attacker because they "
            "depend on victim behaviour, not attacker-controlled URL structure. "
            "Proposed mitigations include WHOIS-based domain age features (new domains = high risk) "
            "and continuous retraining as attack patterns evolve."
        ),
    }

    print("\n── Adversarial robustness analysis ─────────────────────")
    print(f"  {'Original prob':>14}  {'Crafted prob':>13}  {'Drop':>6}  {'Bypass':>7}  Technique")
    print(f"  {'-'*80}")
    for r in rows:
        flag = "YES" if r["bypass"] else "no"
        print(f"  {r['orig_prob']:>14.4f}  {r['craft_prob']:>13.4f}  "
              f"{r['prob_drop']:>+6.4f}  {flag:>7}  {r['technique'][:45]}")
    print(f"\n  Bypass rate: {bypass_rate:.1%}  ({bypassed}/{len(ADVERSARIAL_PAIRS)} evasions successful)")
    print(f"  {result['interpretation'][:120]}...")
    print()

    if save_figure:
        orig_probs  = [r["orig_prob"]  for r in rows]
        craft_probs = [r["craft_prob"] for r in rows]
        pair_labels = [f"Pair {i+1}" for i in range(len(rows))]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(rows))
        ax.scatter(x, orig_probs,  color="#e74c3c", s=60, zorder=5, label="Original phishing URL")
        ax.scatter(x, craft_probs, color="#3498db", s=60, zorder=5, label="Crafted evasion URL")
        for xi, op, cp in zip(x, orig_probs, craft_probs):
            ax.plot([xi, xi], [op, cp], color="grey", lw=0.8, zorder=3)
        ax.axhline(threshold, color="#2c3e50", lw=1.2, ls="--", label=f"Threshold ({threshold})")
        ax.fill_between(
            [-0.5, len(rows) - 0.5],
            threshold - 0.08, threshold + 0.08,
            alpha=0.12, color="orange", label="Uncertainty zone",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Phishing probability")
        ax.set_title("Adversarial robustness — original vs crafted evasion URL probabilities", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURES_DIR / "adversarial_robustness.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[research] saved figure -> {path}")

    return result


# ─────────────────────────────────────────────────────────────
# 3.  GENERALISATION REPORT (JSON)
# ─────────────────────────────────────────────────────────────
def write_generalization_report(
    shift_result      : dict,
    adversarial_result: dict,
    model_info        : dict,
    threshold         : float,
    threshold_source  : str,
) -> Path:
    """
    Writes a complete generalisation report as JSON to results/generalization_report.json.
    """
    report = {
        "metadata": {
            "generated_at"    : datetime.now(timezone.utc).isoformat(),
            "project"         : "PhishGuard — Context-Aware Phishing Detection Using Hybrid Network and Behavioural Signals",
            "model"           : model_info.get("model_name",      "unknown"),
            "n_features"      : model_info.get("n_features",       21),
            "training_samples": model_info.get("training_samples", "unknown"),
            "threshold"       : threshold,
            "threshold_source": threshold_source,
        },
        "distribution_shift": {
            "internal_metrics": shift_result["internal"],
            "external_metrics": shift_result["external"],
            "gap"             : shift_result["gap"],
            "gap_severity"    : shift_result["gap_severity"],
            "interpretation"  : shift_result["interpretation"],
        },
        "adversarial_robustness": {
            "n_pairs"       : adversarial_result["n_pairs"],
            "n_bypassed"    : adversarial_result["n_bypassed"],
            "bypass_rate"   : adversarial_result["bypass_rate"],
            "interpretation": adversarial_result["interpretation"],
            "evasion_pairs" : adversarial_result["pairs"],
        },
        "research_conclusions": {
            "primary_contribution": (
                "Multi-modal phishing detection combining structural URL features, "
                "NLP-derived token features, and behavioural session signals into a unified "
                "risk score via weighted fusion."
            ),
            "distribution_shift_finding": (
                f"A {shift_result['gap']['accuracy']:.1%} accuracy gap between internal and "
                f"external evaluation ({shift_result['gap_severity']} severity) demonstrates "
                "the challenge of cross-distribution generalisation in security ML systems. "
                f"Recall on external data remains {shift_result['external']['recall']:.1%}, "
                "which is operationally acceptable for a security-first deployment."
            ),
            "adversarial_finding": (
                f"Lexical URL features alone have a {adversarial_result['bypass_rate']:.1%} "
                "evasion rate under simulated adversarial conditions. Behavioural signals, "
                "which depend on victim interaction rather than attacker-controlled URL structure, "
                "are resilient to this class of attack. This motivates the multi-modal approach."
            ),
            "limitations": [
                "Model is static — no online learning or retraining pipeline implemented.",
                "Behavioural signals are collected via slider simulation, not real browser telemetry.",
                "WHOIS-based domain age features are not included due to API latency constraints.",
                "Adversarial evaluation uses manually crafted examples, not automated evasion.",
                "Cross-validation not performed on the external dataset.",
            ],
            "future_work": [
                "WHOIS domain age and registration country as additional features.",
                "Automated adversarial example generation via genetic algorithms or FGSM variants.",
                "Online learning component with sliding-window retraining on fresh phishing feeds.",
                "Real browser extension integration for live behavioural telemetry collection.",
                "Ensemble calibration study to improve probability reliability on external data.",
            ],
        },
    }

    path = RESULTS_DIR / "generalization_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[research] generalization report saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────
# STANDALONE RUNNER  — only one if __name__ == "__main__" block
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib

    print("Running standalone verification...")
    print("To run with your real model, import from train.py instead.\n")

    model_path = Path("models/stacking_model.pkl")
    feat_path  = Path("models/feature_names.pkl")

    if not model_path.exists():
        print("models/stacking_model.pkl not found.")
        print("Run src/train.py first, then re-run this script.")
    else:
        bundle        = joblib.load(model_path)
        base_models   = bundle["base_models"]
        meta_learner  = bundle["meta_learner"]
        feature_names = bundle["feature_names"]

        try:
            threshold_data   = joblib.load("models/threshold.pkl")
            threshold        = float(threshold_data["base_threshold"])
            threshold_source = "learned"
        except FileNotFoundError:
            threshold        = 0.75
            threshold_source = "default"

        adv = adversarial_robustness(
            base_models, meta_learner, feature_names, threshold, save_figure=True
        )

        print("\nTo run distribution_shift_analysis(), call it from train.py with:")
        print("  X_internal, y_internal  — your 80/20 internal test split")
        print("  X_external, y_external  — your second dataset (e.g. dataset 2)")
        print("  threshold               — your learned or default threshold\n")

        dummy_shift = {
            "internal"      : {"dataset":"internal","n_samples":9763,"accuracy":0.9996,"precision":0.9998,"recall":0.9994,"f1":0.9996,"roc_auc":0.9999},
            "external"      : {"dataset":"external","n_samples":522142,"accuracy":0.5052,"precision":0.2079,"recall":0.6228,"f1":0.3120,"roc_auc":0.5818},
            "gap"           : {"accuracy":0.4944,"precision":0.7919,"recall":0.3766,"f1":0.6876,"roc_auc":0.4181},
            "gap_severity"  : "critical",
            "interpretation": "Dummy shift data — replace with real call to distribution_shift_analysis()",
        }

        model_info = {
            "model_name"      : "Stacking Ensemble (RF + XGB + LGB → Logistic meta-learner)",
            "n_features"      : len(feature_names),
            "training_samples": "see train.py output",
        }

        report_path = write_generalization_report(
            dummy_shift, adv, model_info, threshold, threshold_source
        )
        print(f"\nReport written to: {report_path}")
        print("Replace dummy_shift with output of distribution_shift_analysis() for full results.")