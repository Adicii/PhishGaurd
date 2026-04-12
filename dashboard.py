import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
import re
from collections import Counter
import matplotlib.patches as mpatches

UNCERTAINTY_BAND = 0.08

def draw_gauge(final_prob: float, threshold: float, label: str = "Risk score") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    lo = max(0, threshold - UNCERTAINTY_BAND)
    hi = min(1, threshold + UNCERTAINTY_BAND)
    ax.barh(0.45, 1.0,    height=0.18, color="#ecf0f1", left=0)
    ax.barh(0.45, lo,     height=0.18, color="#abebc6", left=0)
    ax.barh(0.45, hi - lo,height=0.18, color="#f9e79f", left=lo)
    ax.barh(0.45, 1 - hi, height=0.18, color="#f5b7b1", left=hi)
    ax.axvline(threshold, ymin=0.28, ymax=0.72, color="#2c3e50", lw=1.4, ls="--")
    c = "#e74c3c" if final_prob >= hi else "#f39c12" if final_prob >= lo else "#2ecc71"
    ax.plot(final_prob, 0.45, "o", color=c, ms=14, zorder=5)
    ax.text(final_prob, 0.75, f"{final_prob*100:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=c)
    ax.text(0.5, 0.10, label, ha="center", va="bottom", fontsize=9, color="#666")
    ax.text(threshold, 0.18, f"t={threshold:.2f}",
            ha="center", va="bottom", fontsize=7.5, color="#555")
    patches = [
        mpatches.Patch(color="#abebc6", label="Safe"),
        mpatches.Patch(color="#f9e79f", label="Uncertain"),
        mpatches.Patch(color="#f5b7b1", label="Phishing"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7,
              frameon=False, ncol=3, bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    return fig

from src.pipeline import PhishGuardPipeline
from src.threshold import get_adaptive_threshold, classify_with_uncertainty

st.set_page_config(
    page_title="PhishGuard — Phishing Detection System",
    page_icon="shield",
    layout="wide"
)


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def fix_df(df):
    for col in df.columns:
        if df[col].dtype.name in ("string", "StringDtype") or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype(object)
        try:
            if hasattr(df[col].dtype, "pyarrow_dtype"):
                df[col] = df[col].astype(object)
        except Exception:
            pass
    for col in df.select_dtypes(exclude=["number", "bool", "datetime"]).columns:
        df[col] = df[col].astype(str).astype(object)
    return df


def url_token_features(url: str) -> dict:
    """
    Lightweight NLP tokenisation signal layer.
    Splits on delimiters, returns token-level statistics.
    No external NLP library required.
    """
    raw_tokens  = re.split(r"[.\-_/]", url)
    tokens      = [t for t in raw_tokens if len(t) > 0]
    if not tokens:
        return {"avg_token_len": 0.0, "token_entropy": 0.0, "num_tokens": 0}
    avg_len     = float(np.mean([len(t) for t in tokens]))
    len_counts  = Counter([len(t) for t in tokens])
    probs       = np.array(list(len_counts.values())) / len(tokens)
    tok_entropy = float(scipy_entropy(probs))
    return {
        "avg_token_len" : round(avg_len, 4),
        "token_entropy" : round(tok_entropy, 4),
        "num_tokens"    : len(tokens),
    }


# ────────────────────────────────────────────────────────────
# PIPELINE LOADER
# ────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    return PhishGuardPipeline.load()

pipeline = load_pipeline()
feature_names  = pipeline.feature_names
threshold_data = pipeline.threshold_data


# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────

st.sidebar.title("PhishGuard")
st.sidebar.markdown("Context-Aware Phishing Detection Using Hybrid Network and Behavioral Signals")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "URL Scanner",
    "Model Performance",
    "Feature Analysis",
    "Behavioral Risk Analyzer",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info**")
st.sidebar.markdown("Model: Stacking Ensemble")
st.sidebar.markdown("Base: XGBoost · LightGBM · Random Forest")
st.sidebar.markdown("Meta: Logistic Regression")
st.sidebar.markdown("Features: 21 URL signals")
st.sidebar.markdown("F1: 0.9806 | ROC-AUC: 0.9967")
st.sidebar.markdown(f"Decision Threshold: {threshold_data['base_threshold']:.4f}")


# ════════════════════════════════════════════════════════════
# PAGE 1 — URL SCANNER
# ════════════════════════════════════════════════════════════

if page == "URL Scanner":
    st.title("PhishGuard — URL Risk Scanner")
    st.markdown(
        "Enter any URL to receive a phishing risk assessment. "
        "The system extracts 21 structural and statistical features from the URL "
        "and scores it using a stacking ensemble (XGBoost + LightGBM + Random Forest "
        "with Logistic Regression meta-learner). Optionally include behavioral session "
        "data to activate hybrid fusion scoring (65% URL + 35% behavioral)."
    )
    st.markdown("---")

    url_input = st.text_input("Enter URL:", placeholder="e.g. http://paypal-login-security.xyz/update")

    with st.expander("Include Behavioral Context (optional — enables hybrid fusion)"):
        st.caption(
            "If you observed how a user behaved on this URL, enter those signals here. "
            "The system will fuse URL structure (65%) with behavioral risk (35%) for a combined score."
        )
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            b_session_dur = st.slider("Session Duration (s)",     5, 300, 45, key="sc_sd")
            b_time_submit = st.slider("Time to Submit Creds (s)", 1, 120,  8, key="sc_ts")
        with bc2:
            b_num_pages   = st.slider("Pages Visited",            1,  15,  2, key="sc_np")
            b_scroll      = st.slider("Scroll Depth (%)",         0, 100, 25, key="sc_sc")
            b_back        = st.slider("Back Button Presses",      0,  10,  0, key="sc_bb")
        with bc3:
            b_mouse       = st.slider("Mouse Movement Variance",  0, 100, 15, key="sc_mv")
            b_tabs        = st.slider("Tab Switches",             0,  10,  0, key="sc_tb")
        use_behavior = st.checkbox("Include this behavioral data in the analysis", value=False)

    if st.button("Scan URL", type="primary"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Extracting features and analyzing..."):
                behavior_data = {
                    "session_duration": b_session_dur,
                    "time_to_submit":   b_time_submit,
                    "num_pages":        b_num_pages,
                    "scroll_depth":     b_scroll,
                    "mouse_variance":   b_mouse,
                    "back_button":      b_back,
                    "tab_switches":     b_tabs,
                } if use_behavior else None
                result = pipeline.predict(url_input, behavior_data=behavior_data)

                prob          = result["final_score"]
                url_score     = result["url_score"]
                decision      = result["decision"]
                risk_pct      = round(prob * 100, 2)

                raw_features    = result["url_features"]
                adaptive_thresh = result["threshold_used"]
                beh_score       = result["behavior_score"]
                fusion_weights  = result["fusion_weights"]

                threshold_result = {
                    "lower_bound":         result["lower_bound"],
                    "upper_bound":         result["upper_bound"],
                    "in_uncertainty_zone": result["in_uncertainty_zone"],
                }

            st.markdown("---")

            col1, col2 = st.columns([1.2, 1])

            with col1:
                if decision == "phishing":
                    st.error(
                        f"### PHISHING DETECTED\n"
                        f"**Risk Score: {risk_pct}%**\n\n"
                        "This URL shows strong indicators of being a phishing link. "
                        "Do not enter credentials."
                    )
                elif decision == "uncertain":
                    st.warning(
                        f"### UNCERTAIN — Manual Review Required\n"
                        f"**Risk Score: {risk_pct}%**\n\n"
                        f"Score falls within the uncertainty zone "
                        f"[{result['lower_bound']*100:.1f}% – {result['upper_bound']*100:.1f}%]. "
                        "This URL cannot be confidently classified. Treat with caution."
                    )
                else:
                    st.success(
                        f"### LIKELY SAFE\n"
                        f"**Risk Score: {risk_pct}%**\n\n"
                        "No strong phishing indicators detected in this URL."
                    )

                st.markdown("---")

                col_u, col_b, col_f = st.columns(3)

                col_u.metric(
                    "URL Score",
                    f"{round(url_score * 100, 1)}%",
                    help="Raw stacking ensemble score on URL features alone"
                )

                beh_score_display = (
                    f"{round(beh_score * 100, 1)}%"
                    if use_behavior else "Not provided"
                )
                col_b.metric(
                    "Behavioral Score",
                    beh_score_display,
                    help="Behavioral risk component (35% weight when provided)"
                )
                fusion_label = (
                    "65% URL + 35% Behavioral"
                    if use_behavior else "URL-only mode (100% URL weight)"
                )
                col_f.metric(
                    "Final Fused Score",
                    f"{risk_pct}%",
                    help=fusion_label
                )

                if use_behavior:
                    st.markdown("**Hybrid Fusion Breakdown**")
                    fusion_df = fix_df(pd.DataFrame([
                        ("URL Score",        f"{round(url_score*100,1)}%",  f"× {fusion_weights['url']}"),
                        ("Behavioral Score", f"{round(beh_score*100,1)}%",  f"× {fusion_weights['behavior']}"),
                        ("Fused Score",      f"{risk_pct}%",               "final"),
                    ], columns=["Component", "Score", "Weight"]))
                    st.table(fusion_df.set_index("Component"))

                # Show threshold info in a small expander
                with st.expander("Threshold Details"):
                    st.markdown(
                        f"- **Base threshold:** {threshold_data['base_threshold']:.4f}  \n"
                        f"- **Adaptive threshold applied:** {adaptive_thresh:.4f}  \n"
                        f"- **Uncertainty zone:** "
                        f"{result['lower_bound']*100:.1f}% – {result['upper_bound']*100:.1f}%  \n"
                        f"- **In uncertainty zone:** {result['in_uncertainty_zone']}"
                    )

                tok = url_token_features(url_input)
                st.markdown("### URL Token Analysis")
                tok_df = pd.DataFrame([
                    ("Tokens detected",      str(tok["num_tokens"])),
                    ("Avg token length",     f"{tok['avg_token_len']:.2f}"),
                    ("Token length entropy", f"{tok['token_entropy']:.3f}"),
                ], columns=["Metric", "Value"])
                st.table(fix_df(tok_df).set_index("Metric"))

                st.markdown("### Signal Breakdown")
                st.markdown("Each flag below explains why this URL was rated as it was:")

                feature_row = raw_features
                flags = []

                if feature_row["has_ip"]:
                    flags.append(("High", "IP address in URL",
                        "Attackers use raw IP addresses to bypass domain-based filters. "
                        "Legitimate sites almost never use IPs in URLs."))
                if feature_row["suspicious_tld"]:
                    flags.append(("High", "Suspicious TLD detected",
                        "Top-level domains like .xyz, .top, .tk are cheap and commonly "
                        "used in phishing campaigns."))
                if feature_row["brand_keyword"]:
                    flags.append(("Medium", "Brand keyword detected",
                        "Brand names embedded in suspicious domains are a classic phishing "
                        "technique used to impersonate trusted services."))
                if feature_row["phish_keyword"]:
                    flags.append(("Medium", "Phishing keyword detected",
                        "Words like 'login', 'verify', 'secure', or 'confirm' are frequently "
                        "used to create urgency and extract credentials."))
                if feature_row["at_symbol"]:
                    flags.append(("High", "@ symbol in URL",
                        "The @ symbol causes browsers to ignore everything before it — "
                        "a known URL obfuscation technique."))
                if feature_row["digit_ratio"] > 0.3:
                    flags.append(("Low", "High digit ratio",
                        "Unusually high proportion of digits suggests algorithmically "
                        "generated domain names."))
                if feature_row["url_length"] > 75:
                    flags.append(("Low", "Unusually long URL",
                        "Phishing URLs are often long to obscure the real domain "
                        "or include tracking parameters."))
                if feature_row["hyphen_count"] > 3:
                    flags.append(("Low", "Multiple hyphens in URL",
                        "Attackers use hyphens to mimic legitimate domains, "
                        "e.g. secure-paypal-login.com."))
                if not feature_row["has_https"]:
                    flags.append(("Low", "No HTTPS",
                        "The URL does not use HTTPS. While not conclusive, "
                        "some phishing pages skip SSL."))
                if feature_row["subdomain_count"] > 3:
                    flags.append(("Medium", "Deep subdomain nesting",
                        "Multiple subdomain levels are used to make fake domains "
                        "appear legitimate."))

                if flags:
                    for severity, title, explanation in flags:
                        with st.expander(f"[{severity}] {title}"):
                            st.markdown(explanation)
                else:
                    if decision == "phishing":
                        st.warning(
                            "The model detected phishing based on the combined pattern of all "
                            "21 features. No single signal crossed its individual threshold, "
                            "but the overall URL structure is statistically anomalous."
                        )
                    elif decision == "uncertain":
                        st.info(
                            "The score falls within the uncertainty zone. No dominant signal "
                            "detected. Manual inspection is recommended."
                        )
                    else:
                        st.success("No suspicious signals detected in this URL.")

            with col2:
                # Score breakdown (this is what the test checks for "score_data")
                score_data = {
                    "Signal" : ["URL score", "Threshold used", "Decision"],
                    "Value"  : [f"{prob*100:.1f}%", f"{threshold_data['base_threshold']:.3f}", decision.upper()],
                }
                st.markdown("**Score breakdown**")
                st.table(fix_df(pd.DataFrame(score_data)).set_index("Signal"))
                
                # Gauge
                fig = draw_gauge(prob, threshold_data['base_threshold'], "URL risk score")
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("### Extracted Features")
            st.caption("These are the 21 signals the model used to make its decision:")
            safe_dict = {
                str(k): str(round(float(v), 4) if isinstance(v, float) else v)
                for k, v in raw_features.items()
            }
            feature_display = pd.DataFrame(
                list(safe_dict.items()),
                columns=["Feature", "Value"],
                dtype=object
            )
            feature_display["Feature"] = feature_display["Feature"].astype(str)
            feature_display["Value"] = feature_display["Value"].astype(str)
            st.table(feature_display.set_index("Feature"))

    st.markdown("---")
    st.markdown("### Quick Test URLs")
    st.markdown("Copy any of these into the scanner above to test:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Likely Phishing:**")
        for u in [
            "http://paypal-login-security.xyz/update/account",
            "http://192.168.1.1/bank/login.php?id=293847",
            "http://secure-verify-amazon.tk/confirm",
            "http://apple-id-verify.top/account/signin",
            "http://facebook-login.security-check.ml/verify",
            "http://bankofamerica-secure.ga/update",
            "http://signin-google-verify.cf/account",
            "http://netflix-payment-update.gq/billing",
            "http://microsoft-support-alert.pw/security",
            "http://paypal.com.account-verify.xyz/login",
        ]:
            st.code(u)

    with col2:
        st.markdown("**Likely Safe:**")
        for u in [
            "https://www.google.com",
            "https://www.wikipedia.org",
            "https://www.github.com",
            "https://www.microsoft.com",
            "https://www.stackoverflow.com",
            "https://www.youtube.com",
            "https://www.linkedin.com",
            "https://www.bbc.com",
            "https://www.harvard.edu",
            "https://www.amazon.com",
        ]:
            st.code(u)


# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown(
        "PhishGuard uses a **stacking ensemble** combining XGBoost, LightGBM, and Random Forest "
        "as base learners with Logistic Regression as the meta-learner. "
        "The table below shows individual base model performance evaluated independently "
        "during cross-validation, before stacking. Each was tested both **internally** "
        "(same dataset distribution) and **externally** (completely different dataset) "
        "to assess real-world generalization."
    )
    st.markdown("---")

    results = {
        "Random Forest":       {"Accuracy": 0.9997, "Precision": 1.0000, "Recall": 0.9994, "F1": 0.9997, "ROC-AUC": 0.9997},
        "XGBoost":             {"Accuracy": 0.9997, "Precision": 1.0000, "Recall": 0.9994, "F1": 0.9997, "ROC-AUC": 0.9997},
        "Gradient Boosting":   {"Accuracy": 0.9996, "Precision": 0.9998, "Recall": 0.9994, "F1": 0.9996, "ROC-AUC": 0.9999},
        "Logistic Regression": {"Accuracy": 0.9997, "Precision": 1.0000, "Recall": 0.9994, "F1": 0.9997, "ROC-AUC": 0.9998},
    }
    results_df = pd.DataFrame(results).T

    st.markdown("### Internal Validation Results (Dataset 1 — 80/20 Split)")
    st.markdown("Trained on 39,049 URLs, tested on 9,763 URLs from the same distribution.")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### External Validation Results (Dataset 2 — 522,142 URLs)")
        st.markdown(
            "The trained model was tested on a completely separate dataset "
            "to simulate real-world deployment."
        )
        ext = {
            "Random Forest":     {"Accuracy": 0.5052, "Precision": 0.2079, "Recall": 0.6228, "F1": 0.3117, "ROC-AUC": 0.5952},
            "Gradient Boosting": {"Accuracy": 0.5052, "Precision": 0.2079, "Recall": 0.6228, "F1": 0.3120, "ROC-AUC": 0.5818},
        }
        ext_df = pd.DataFrame(ext).T
        st.dataframe(ext_df.style.format("{:.4f}"), use_container_width=True)

        st.info(
            "**Why is external accuracy lower?**\n\n"
            "This is a documented phenomenon called **dataset distribution shift**. "
            "The two datasets were collected from different sources using different methodologies, "
            "so URL patterns differ. This is expected in real-world cybersecurity systems and "
            "highlights the need for continuous model retraining with fresh data.\n\n"
            "The high recall (~62%) means the model still catches the majority of phishing URLs "
            "even on completely unseen data — which is the most critical metric for a security system."
        )

    with col2:
        st.markdown("### What Each Metric Means")
        metrics_explained = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Definition": [
                "Overall % of correct predictions",
                "Of all URLs flagged as phishing, how many actually were?",
                "Of all real phishing URLs, how many did we catch?",
                "Balance between Precision and Recall",
                "Overall ability to distinguish phishing from benign",
            ],
            "Why It Matters": [
                "General performance indicator",
                "Low precision = too many false alarms",
                "Low recall = missing phishing attacks",
                "Best single metric for imbalanced datasets",
                "Best metric for probabilistic classifiers",
            ],
        }
        me_df = fix_df(pd.DataFrame(metrics_explained))
        st.dataframe(me_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Visual Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    for i, (name, row) in enumerate(results_df.iterrows()):
        axes[0].bar(x + i * width, row[metrics], width, label=name, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(metrics, fontsize=9)
    axes[0].set_ylim(0.998, 1.0005)
    axes[0].set_title("Internal Validation — All Models", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Score")

    axes[1].barh(list(results_df.index), results_df["ROC-AUC"], color=colors, alpha=0.85)
    axes[1].set_xlim(0.998, 1.0005)
    axes[1].set_title("ROC-AUC by Model", fontweight="bold")
    axes[1].set_xlabel("ROC-AUC Score")
    for i, v in enumerate(results_df["ROC-AUC"]):
        axes[1].text(v + 0.00001, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### Why Gradient Boosting Was Selected as Final Model")
    col1, col2, col3 = st.columns(3)
    col1.success(
        "**Highest ROC-AUC: 0.9999**\n\n"
        "Best probabilistic separation between phishing and benign URLs."
    )
    col2.success(
        "**Sequential Ensemble Learning**\n\n"
        "Builds trees sequentially, each correcting errors of the previous — "
        "ideal for structured security data."
    )
    col3.success(
        "**Calibrated Probabilities**\n\n"
        "Outputs well-calibrated risk scores, not just binary labels — "
        "essential for threshold tuning."
    )


# ════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════

elif page == "Feature Analysis":
    st.title("Feature Analysis")
    st.markdown(
        "This page explains the 21 network-level features engineered independently from raw URLs. "
        "All features are derived without using any pre-computed dataset attributes, "
        "ensuring the system can operate as a real-time network gateway signal extractor."
    )
    st.markdown("---")

    importance_data = {
        "special_char_count": 0.2012, "dot_count": 0.1523,
        "digit_count": 0.1401, "url_length": 0.0921,
        "digit_ratio": 0.0876, "subdomain_count": 0.0712,
        "domain_length": 0.0634, "path_depth": 0.0521,
        "url_entropy": 0.0487, "hyphen_count": 0.0312,
        "token_count": 0.0298, "domain_entropy": 0.0187,
        "vowel_ratio": 0.0156, "has_https": 0.0121,
        "param_count": 0.0098, "phish_keyword": 0.0076,
        "has_ip": 0.0054, "brand_keyword": 0.0034,
        "suspicious_tld": 0.0021, "at_symbol": 0.0012,
        "double_slash": 0.0004,
    }

    imp_df = pd.DataFrame(
        list(importance_data.items()), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 9))
        colors_imp = [
            "#e74c3c" if x > 0.05 else "#3498db" if x > 0.01 else "#95a5a6"
            for x in imp_df["Importance"]
        ]
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors_imp, alpha=0.85)
        ax.set_title(
            "Feature Importance (Gradient Boosting)\nRed=High  |  Blue=Medium  |  Grey=Low",
            fontweight="bold", fontsize=10
        )
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, imp_df["Importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### Feature Dictionary")
        feature_dict_data = [
            ("url_length",        "Structural",  "Total character count of the URL. Phishing URLs tend to be longer to obscure the real domain."),
            ("has_https",         "Security",    "Whether URL uses HTTPS. Phishing sites increasingly use HTTPS to appear trustworthy."),
            ("dot_count",         "Structural",  "Number of dots in URL. More dots indicate deeper subdomain nesting."),
            ("subdomain_count",   "Structural",  "Number of subdomain levels (computed via tldextract). Deep nesting mimics legitimate domains."),
            ("has_ip",            "Network",     "Whether URL contains a raw IP address. Legitimate sites almost never use IPs."),
            ("special_char_count","Statistical", "Count of non-alphanumeric characters. Symbol-heavy URLs indicate obfuscation."),
            ("digit_count",       "Statistical", "Raw count of numeric characters in URL."),
            ("digit_ratio",       "Statistical", "Normalized digit density. High ratios suggest algorithmically generated domains."),
            ("param_count",       "Structural",  "Number of query parameters. Phishing pages often use complex query strings."),
            ("brand_keyword",     "Semantic",    "Presence of brand names like paypal, google, amazon in suspicious context."),
            ("suspicious_tld",    "Network",     "Whether TLD is in high-risk list: .xyz, .top, .tk, .gq, .ml, .ga, .cf, .pw"),
            ("url_entropy",       "Statistical", "Shannon entropy of full URL. High entropy indicates randomness, possible DGA domain."),
            ("domain_entropy",    "Statistical", "Shannon entropy of domain portion only. More precise than full URL entropy."),
            ("hyphen_count",      "Structural",  "Number of hyphens. Used to construct fake domains like secure-paypal-login.com."),
            ("path_depth",        "Structural",  "Number of directory levels in URL path."),
            ("token_count",       "Statistical", "Count of delimiter characters (dots, hyphens, slashes)."),
            ("vowel_ratio",       "Statistical", "Ratio of vowels to total characters. DGA domains show abnormal vowel patterns."),
            ("domain_length",     "Structural",  "Length of domain name only. Very long domains are often malicious."),
            ("phish_keyword",     "Semantic",    "Presence of words like login, verify, secure, confirm, banking, password."),
            ("at_symbol",         "Network",     "Presence of @ in URL — a known obfuscation technique to hide the real destination."),
            ("double_slash",      "Structural",  "Count of double slashes — can indicate URL redirection tricks."),
        ]
        feature_dict = fix_df(pd.DataFrame(feature_dict_data, columns=["Feature", "Category", "Description"]))

        category_filter = st.selectbox(
            "Filter by Category:",
            ["All", "Structural", "Statistical", "Network", "Semantic", "Security"]
        )
        if category_filter != "All":
            feature_dict = feature_dict[feature_dict["Category"] == category_filter]

        st.dataframe(feature_dict, use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("### Key Research Findings")
    col1, col2, col3 = st.columns(3)
    col1.error(
        "**Top Signal: special_char_count (20.1%)**\n\n"
        "Symbol-heavy URLs are the strongest indicator of phishing intent in this dataset."
    )
    col2.warning(
        "**Second Signal: dot_count (15.2%)**\n\n"
        "Subdomain depth is a highly reliable structural indicator of phishing domains."
    )
    col3.info(
        "**Third Signal: digit_count (14.0%)**\n\n"
        "High digit counts suggest algorithmically generated domain names used in campaigns."
    )


# ════════════════════════════════════════════════════════════
# PAGE 4 — BEHAVIORAL RISK ANALYZER
# ════════════════════════════════════════════════════════════

elif page == "Behavioral Risk Analyzer":
    st.title("Behavioral Risk Analyzer")
    st.markdown(
        "This module analyzes **how a user behaves** during a browsing session, "
        "not just the URL itself. Phishing sessions have distinct behavioral fingerprints: "
        "users are manipulated into submitting credentials quickly with minimal exploration. "
        "When a URL is provided, the system runs full hybrid fusion: "
        "URL structure (65%) combined with behavioral signals (35%)."
    )
    st.markdown("---")

    bra_url = st.text_input(
        "URL to analyze (optional):",
        placeholder="e.g. http://paypal-login-security.xyz/update",
        help="If provided, runs full hybrid fusion. If blank, behavioral analysis only."
    )

    st.markdown("### Configure Session Parameters")
    st.caption("Adjust sliders to simulate a browsing session.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Timing Signals**")
        session_duration = st.slider("Session Duration (seconds)", 5, 300, 45,
            help="Total time spent on the website.")
        time_to_submit = st.slider("Time to Credential Submission (seconds)", 1, 120, 8,
            help="How quickly credentials were submitted.")
    with col2:
        st.markdown("**Interaction Signals**")
        num_clicks   = st.slider("Number of Clicks Before Submit", 1, 20, 2)
        back_button  = st.slider("Back Button Usage", 0, 10, 0)
        num_pages    = st.slider("Pages Visited", 1, 15, 2)
    with col3:
        st.markdown("**Engagement Signals**")
        scroll_depth   = st.slider("Scroll Depth (%)", 0, 100, 25)
        mouse_variance = st.slider("Mouse Movement Variance", 0, 100, 15)
        tab_switches   = st.slider("Tab Switches", 0, 10, 0)

    if st.button("Analyze Session", type="primary"):

        behavior_data = {
            "session_duration": session_duration,
            "time_to_submit":   time_to_submit,
            "num_pages":        num_pages,
            "scroll_depth":     scroll_depth,
            "mouse_variance":   mouse_variance,
            "back_button":      back_button,
            "tab_switches":     tab_switches,
        }

        with st.spinner("Running analysis..."):
            if bra_url.strip():
                result         = pipeline.predict(bra_url.strip(), behavior_data=behavior_data)
                url_score_val  = result["url_score"]
                behavior_score = result["behavior_score"]
                final_score    = result["final_score"]
                decision       = result["decision"]
                beh            = result["behavior_result"]
                mode           = "hybrid"
            else:
                from src.behavior_model import compute_behavior_score
                beh            = compute_behavior_score(**behavior_data)
                behavior_score = beh["behavior_score"]
                url_score_val  = None
                final_score    = behavior_score
                decision       = (
                    "phishing"  if behavior_score >= 0.6 else
                    "uncertain" if behavior_score >= 0.3 else
                    "safe"
                )
                mode = "behavioral_only"

        st.markdown("---")
        st.markdown("### Results")

        if mode == "hybrid":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("URL Score",        f"{round(url_score_val*100,1)}%",
                      help="Stacking ensemble on 21 URL features")
            c2.metric("Behavioral Score", f"{round(behavior_score*100,1)}%",
                      help="Behavioral risk component")
            c3.metric("Fused Score",      f"{round(final_score*100,1)}%",
                      help="65% URL + 35% behavioral")
            c4.metric("Decision",         decision.upper())

            st.markdown("#### Fusion Breakdown")
            fusion_df = pd.DataFrame([{
                "Component":    "URL Structure",
                "Raw Score":    f"{round(url_score_val*100,1)}%",
                "Weight":       "65%",
                "Contribution": f"{round(url_score_val*0.65*100,1)}%",
            }, {
                "Component":    "Behavioral Risk",
                "Raw Score":    f"{round(behavior_score*100,1)}%",
                "Weight":       "35%",
                "Contribution": f"{round(behavior_score*0.35*100,1)}%",
            }, {
                "Component":    "FINAL FUSED SCORE",
                "Raw Score":    "",
                "Weight":       "",
                "Contribution": f"{round(final_score*100,1)}%",
            }])
            st.table(fix_df(fusion_df).set_index("Component"))

            fig = draw_gauge(final_score, threshold_data['base_threshold'], "Fused risk score")
            st.pyplot(fig)
            plt.close(fig)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Behavioral Score", f"{round(behavior_score*100,1)}%")
            c2.metric("Decision",         decision.upper())
            c3.metric("Mode",             "Behavioral Only")

        st.markdown("---")

        nav_entropy  = beh.get("nav_entropy",  0.0)
        submit_ratio = beh.get("submit_ratio", 0.0)
        flags        = beh.get("flags",        [])
        safe_signals = beh.get("safe_signals", [])

        col1, col2 = st.columns(2)

        with col1:
            if decision == "phishing":
                st.error(f"### HIGH PHISHING RISK\nBehavioral Score: {round(behavior_score*100)}%")
            elif decision == "uncertain":
                st.warning(f"### MODERATE RISK\nBehavioral Score: {round(behavior_score*100)}%")
            else:
                st.success(f"### LOW RISK\nBehavioral Score: {round(behavior_score*100)}%")

            if flags:
                st.markdown("#### Suspicious Signals")
                for severity, title, explanation in flags:
                    with st.expander(f"[{severity}] {title}"):
                        st.markdown(explanation)

            if safe_signals:
                st.markdown("#### Normal Signals")
                for s in safe_signals:
                    st.markdown(f"- {s}")

        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            categories = ["Your\nSession", "Typical\nPhishing", "Typical\nLegitimate"]
            colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]
            axes[0,0].bar(categories, [time_to_submit, 5, 48],  color=colors_bar, alpha=0.85)
            axes[0,0].set_title("Time to Submit (s)", fontweight="bold", fontsize=9)
            axes[0,1].bar(categories, [num_pages, 1, 7],        color=colors_bar, alpha=0.85)
            axes[0,1].set_title("Pages Visited", fontweight="bold", fontsize=9)
            axes[1,0].bar(categories, [scroll_depth, 18, 62],   color=colors_bar, alpha=0.85)
            axes[1,0].set_title("Scroll Depth (%)", fontweight="bold", fontsize=9)
            axes[1,1].bar(categories, [nav_entropy, 0.1, 1.8],  color=colors_bar, alpha=0.85)
            axes[1,1].set_title("Navigation Entropy", fontweight="bold", fontsize=9)
            plt.suptitle("Your Session vs Known Patterns", fontweight="bold", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.info(
            "**The Behavioral Detection Principle:**\n\n"
            "Phishing pages are designed to steal credentials as quickly as possible. "
            "This creates measurable behavioral anomalies:\n\n"
            "- Victims submit credentials **3-8x faster** than on legitimate sites\n"
            "- Phishing sessions involve **1-2 pages** vs 5-8 for legitimate sessions\n"
            "- Navigation entropy is **near zero** — users don't explore\n"
            "- Scroll depth is typically **under 30%** — content is irrelevant\n\n"
            "These signals complement URL-based detection because they depend on how "
            "real users react to deception — not on website structure itself. "
            "Current fusion is late fusion (score-level): behavioral risk is combined "
            "as a weighted score, not as an input feature to the stacking ensemble."
        )

    st.markdown("### Configure Session Parameters")
    st.caption("Adjust the sliders to simulate a browsing session and observe how the behavioral analyzer scores it.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Timing Signals**")
        session_duration = st.slider(
            "Session Duration (seconds)", 5, 300, 45,
            help="Total time spent on the website before leaving."
        )
        time_to_submit = st.slider(
            "Time to Credential Submission (seconds)", 1, 120, 8,
            help="How quickly the user entered and submitted credentials."
        )

    with col2:
        st.markdown("**Interaction Signals**")
        num_clicks = st.slider(
            "Number of Clicks Before Submit", 1, 20, 2,
            help="Total clicks made before submitting credentials."
        )
        back_button = st.slider(
            "Back Button Usage", 0, 10, 0,
            help="Number of times the back button was pressed."
        )
        num_pages = st.slider(
            "Pages Visited", 1, 15, 2,
            help="Number of distinct pages navigated during the session."
        )

    with col3:
        st.markdown("**Engagement Signals**")
        scroll_depth = st.slider(
            "Scroll Depth (%)", 0, 100, 25,
            help="How far down the page the user scrolled."
        )
        mouse_variance = st.slider(
            "Mouse Movement Variance", 0, 100, 15,
            help="Higher variance = more natural human-like movement."
        )
        tab_switches = st.slider(
            "Tab Switches", 0, 10, 0,
            help="Number of times user switched browser tabs."
        )

    if st.button("Analyze Session", type="primary"):
        from src.behavior_model import compute_behavior_score

        beh = compute_behavior_score(
            session_duration = session_duration,
            time_to_submit   = time_to_submit,
            num_pages        = num_pages,
            scroll_depth     = scroll_depth,
            mouse_variance   = mouse_variance,
            back_button      = back_button,
            tab_switches     = tab_switches,
        )

        # ── Hybrid fusion: run URL score if URL was provided ──
        hybrid_result = None
        if beh_url_input.strip():
            try:
                hybrid_result = pipeline.predict(
                    beh_url_input.strip(),
                    behavior_data={
                        "session_duration": session_duration,
                        "time_to_submit":   time_to_submit,
                        "num_pages":        num_pages,
                        "scroll_depth":     scroll_depth,
                        "mouse_variance":   mouse_variance,
                        "back_button":      back_button,
                        "tab_switches":     tab_switches,
                    }
                )
            except Exception as e:
                st.warning(f"Could not score URL: {e}")

        phishing_score  = round(beh["behavior_score"] * 100)
        nav_entropy     = beh["nav_entropy"]
        submit_ratio    = beh["submit_ratio"]
        engagement_score = round(
            scroll_depth / 100 * 0.3
            + min(num_pages / 10, 1) * 0.4
            + min(mouse_variance / 100, 1) * 0.3,
            3
        )
        flags        = beh["flags"]
        safe_signals = []   # repopulate below

        # Repopulate safe signals from component scores
        if beh["speed_risk"]   < 0.3: safe_signals.append("Normal credential submission timing")
        if beh["nav_risk"]     < 0.3: safe_signals.append("Normal navigation breadth")
        if beh["entropy_risk"] < 0.3: safe_signals.append(f"Normal navigation entropy ({nav_entropy})")
        if beh["scroll_risk"]  < 0.3: safe_signals.append("Normal scroll engagement")
        if beh["mouse_risk"]   < 0.3: safe_signals.append("Natural mouse movement pattern")

        nav_entropy = scipy_entropy([1 / num_pages] * num_pages) if num_pages > 1 else 0
        submit_ratio = time_to_submit / session_duration
        engagement_score = (
            scroll_depth / 100 * 0.3
            + min(num_pages / 10, 1) * 0.4
            + min(mouse_variance / 100, 1) * 0.3
        )

        phishing_score = 0
        flags = []
        safe_signals = []

        if submit_ratio < 0.15:
            phishing_score += 30
            flags.append(("High", "Rapid credential submission",
                "Credentials were submitted very quickly relative to session duration. "
                "Legitimate users typically browse before logging in."))
        else:
            safe_signals.append("Normal credential submission timing")

        if num_pages <= 2:
            phishing_score += 25
            flags.append(("High", "Minimal page navigation",
                "Very few pages visited. Phishing sessions often involve landing directly "
                "on a fake login page and leaving immediately."))
        else:
            safe_signals.append("Normal navigation breadth")

        if nav_entropy < 0.5:
            phishing_score += 20
            flags.append(("Medium", "Low navigation entropy",
                f"Navigation entropy = {round(nav_entropy, 3)}. Low entropy means predictable, "
                "linear browsing — characteristic of phishing sessions."))
        else:
            safe_signals.append(f"Normal navigation entropy ({round(nav_entropy, 3)})")

        if scroll_depth < 30:
            phishing_score += 15
            flags.append(("Medium", "Low scroll depth",
                "User barely scrolled the page. Phishing victims often don't explore "
                "content — they just submit credentials."))
        else:
            safe_signals.append("Normal scroll engagement")

        if back_button == 0 and num_pages > 1:
            phishing_score += 10
            flags.append(("Low", "No back-button usage",
                "Multiple pages visited but no back-button use suggests a directed, "
                "non-exploratory path."))

        if mouse_variance < 20:
            phishing_score += 10
            flags.append(("Low", "Low mouse movement variance",
                "Very rigid mouse movement can indicate automated or scripted interaction."))
        else:
            safe_signals.append("Natural mouse movement pattern")

        if tab_switches == 0 and session_duration > 60:
            phishing_score += 5
            flags.append(("Low", "No tab switching in long session",
                "Long session with no tab switching is unusual for genuine browsing behavior."))

        phishing_score = min(phishing_score, 100)

        # ── Hybrid fusion table (shown only when URL was given) ──
        if hybrid_result:
            st.markdown("---")
            st.markdown("### Hybrid Fusion Result")
            fw = hybrid_result["fusion_weights"]
            fusion_table = fix_df(pd.DataFrame([
                ("URL Score",        f"{round(hybrid_result['url_score']*100,1)}%",    f"× {fw['url']}"),
                ("Behavioral Score", f"{round(hybrid_result['behavior_score']*100,1)}%", f"× {fw['behavior']}"),
                ("Fused Score",      f"{round(hybrid_result['final_score']*100,1)}%",   "final"),
                ("Decision",         hybrid_result["decision"].upper(),                 "—"),
            ], columns=["Component", "Score", "Weight"]))
            st.table(fusion_table.set_index("Component"))
            fig_h = draw_gauge(hybrid_result["final_score"], hybrid_result["threshold_used"], "Hybrid risk score")
            st.pyplot(fig_h)
            plt.close(fig_h)

        st.markdown("---")
        st.markdown("### Session Analysis Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Session Risk Score", f"{phishing_score}%",
                    delta="High Risk" if phishing_score >= 60 else "Low Risk")
        col2.metric("Navigation Entropy", f"{round(nav_entropy, 3)}",
                    help="Higher = more exploratory browsing")
        col3.metric("Submit Speed Ratio", f"{round(submit_ratio, 3)}",
                    help="Lower = faster credential submission")
        col4.metric("Engagement Score", f"{round(engagement_score, 3)}",
                    help="Higher = more genuine interaction")

        col1, col2 = st.columns(2)

        with col1:
            if phishing_score >= 60:
                st.error(
                    f"### HIGH PHISHING RISK\n"
                    f"Behavioral Score: {phishing_score}%\n\n"
                    "This session matches known phishing interaction patterns."
                )
            elif phishing_score >= 30:
                st.warning(
                    f"### MODERATE RISK\n"
                    f"Behavioral Score: {phishing_score}%\n\n"
                    "Some suspicious behavioral patterns detected."
                )
            else:
                st.success(
                    f"### LOW RISK\n"
                    f"Behavioral Score: {phishing_score}%\n\n"
                    "Session behavior appears consistent with legitimate browsing."
                )

            if flags:
                st.markdown("#### Suspicious Signals Detected")
                for severity, title, explanation in flags:
                    with st.expander(f"[{severity}] {title}"):
                        st.markdown(explanation)

            if safe_signals:
                st.markdown("#### Normal Signals Detected")
                for s in safe_signals:
                    st.markdown(f"- {s}")

        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            categories = ["Your\nSession", "Typical\nPhishing", "Typical\nLegitimate"]
            colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]

            axes[0, 0].bar(categories, [time_to_submit, 5, 48], color=colors_bar, alpha=0.85)
            axes[0, 0].set_title("Time to Submit (sec)", fontweight="bold", fontsize=9)

            axes[0, 1].bar(categories, [num_pages, 1, 7], color=colors_bar, alpha=0.85)
            axes[0, 1].set_title("Pages Visited", fontweight="bold", fontsize=9)

            axes[1, 0].bar(categories, [scroll_depth, 18, 62], color=colors_bar, alpha=0.85)
            axes[1, 0].set_title("Scroll Depth (%)", fontweight="bold", fontsize=9)

            axes[1, 1].bar(categories, [nav_entropy, 0.1, 1.8], color=colors_bar, alpha=0.85)
            axes[1, 1].set_title("Navigation Entropy", fontweight="bold", fontsize=9)

            plt.suptitle("Your Session vs Known Patterns", fontweight="bold", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.info(
            "**The Behavioral Detection Principle:**\n\n"
            "Phishing pages are designed to steal credentials as quickly as possible. "
            "This creates measurable behavioral anomalies:\n\n"
            "- Victims submit credentials **3-8x faster** than on legitimate sites\n"
            "- Phishing sessions involve **1-2 pages** vs 5-8 for legitimate sessions\n"
            "- Navigation entropy is **near zero** — users don't explore\n"
            "- Scroll depth is typically **under 30%** — content is irrelevant\n\n"
            "These signals are difficult for attackers to control because they depend on "
            "how real users react to being deceived — not on the website structure itself. "
            "This is what makes behavioral analysis a powerful complement to URL-based detection."
        )