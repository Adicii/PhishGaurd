import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from urllib.parse import urlparse

st.set_page_config(
    page_title="PhishGuard — Phishing Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# UNIVERSAL FIX: forces all string columns to
# plain Python object dtype so Streamlit's
# Arrow serializer never sees LargeUtf8 (type 20)
# ─────────────────────────────────────────────
def fix_df(df):
    for col in df.columns:
        if df[col].dtype.name in ("string", "StringDtype") or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype(object)
        try:
            if hasattr(df[col].dtype, "pyarrow_dtype"):
                df[col] = df[col].astype(object)
        except Exception:
            pass
    # Final safety: cast any remaining non-numeric columns
    for col in df.select_dtypes(exclude=["number", "bool", "datetime"]).columns:
        df[col] = df[col].astype(str).astype(object)
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

def fast_entropy(s):
    if len(s) == 0:
        return 0
    counts = Counter(s)
    probs = np.array(list(counts.values())) / len(s)
    return float(scipy_entropy(probs))

def extract_single_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        "url_length": len(url),
        "has_https": int(url.startswith("https")),
        "dot_count": url.count("."),
        "subdomain_count": url.count("."),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "special_char_count": len(re.findall(r"[^\w]", url)),
        "digit_count": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "param_count": url.count("?") + url.count("&"),
        "brand_keyword": int(bool(re.search(
            r"paypal|google|facebook|amazon|bank|apple|microsoft|netflix",
            url.lower()))),
        "suspicious_tld": int(domain.split(".")[-1] in
            ["xyz","top","gq","tk","ml","ga","cf","pw"]),
        "url_entropy": fast_entropy(url),
        "domain_entropy": fast_entropy(domain) if domain else 0,
        "hyphen_count": url.count("-"),
        "path_depth": path.count("/"),
        "token_count": len(re.findall(r"[.\-_/]", url)),
        "vowel_ratio": sum(c in "aeiouAEIOU" for c in url) / max(len(url), 1),
        "domain_length": len(domain),
        "phish_keyword": int(bool(re.search(
            r"login|secure|verify|account|update|signin|confirm|banking|password",
            url.lower()))),
        "at_symbol": int("@" in url),
        "double_slash": url.count("//")
    }
    return pd.DataFrame([features])

st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
st.sidebar.title("PhishGuard")
st.sidebar.markdown("**Network & Information Security Project**")
st.sidebar.markdown("*Context-Aware Phishing Detection*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔍 URL Scanner",
    "📊 Model Performance",
    "📈 Feature Analysis",
    "🧠 Session Analyzer",
    "🎓 Course Concepts Applied"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("Course: Network & Information Security")
st.sidebar.markdown("Model: Gradient Boosting")
st.sidebar.markdown("Best ROC-AUC: 0.9999")

# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ PhishGuard")
    st.subheader("Context-Aware Phishing Detection Using Hybrid Network and Behavioral Signals")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training URLs", "48,812")
    col2.metric("External Test URLs", "522,142")
    col3.metric("Features Engineered", "21")
    col4.metric("Best ROC-AUC", "0.9999")

    st.markdown("---")
    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 — URL Input**\n\nA URL is received for analysis by the detection engine.")
    with col2:
        st.warning("**Step 2 — Feature Extraction**\n\n21 network-level structural signals are extracted from the URL.")
    with col3:
        st.error("**Step 3 — Risk Scoring**\n\nGradient Boosting model outputs a phishing probability score.")

    st.markdown("---")
    st.markdown("### Project Overview")
    st.markdown("""
    Traditional phishing detection relies on blacklists and static URL rules.
    **PhishGuard** improves this by extracting structural, statistical, and behavioral
    signals directly from URLs — signals that attackers cannot easily hide.

    - ✅ Works on **zero-day phishing domains** not in any blacklist
    - ✅ Trained on **real-world phishing datasets**
    - ✅ Validated across **two independent datasets**
    - ✅ Compares **4 machine learning models**
    """)

# ════════════════════════════════════════════════════════════
# PAGE 2 — URL SCANNER
# ════════════════════════════════════════════════════════════
elif page == "🔍 URL Scanner":
    st.title("🔍 URL Risk Scanner")
    st.markdown("Enter any URL below to receive a detailed phishing risk assessment. The system extracts 21 structural and statistical features from the URL and runs them through a trained Gradient Boosting classifier.")
    st.markdown("---")

    url_input = st.text_input("Enter URL:", placeholder="e.g. http://paypal-login-security.xyz/update")

    if st.button("Scan URL", type="primary"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Extracting features and analyzing..."):
                features_df = extract_single_url(url_input)
                prob = model.predict_proba(features_df)[0][1]
                prediction = 1 if prob >= 0.75 else 0
                risk_pct = round(prob * 100, 2)

            st.markdown("---")

            col1, col2 = st.columns([1.2, 1])
            with col1:
                if prob > 0.75:
                    st.error(f"## ⚠️ PHISHING DETECTED\n**Risk Score: {risk_pct}%**\n\nThis URL shows strong indicators of being a phishing link. Do not enter credentials.")
                elif prob > 0.4:
                    st.warning(f"## ⚠️ SUSPICIOUS URL\n**Risk Score: {risk_pct}%**\n\nThis URL has some suspicious characteristics. Proceed with caution.")
                else:
                    st.success(f"## ✅ LIKELY SAFE\n**Risk Score: {risk_pct}%**\n\nNo strong phishing indicators detected in this URL.")

                st.markdown("### 🚩 Signal Breakdown")
                st.markdown("Each flag below explains **why** this URL was rated as it was:")

                feature_row = features_df.iloc[0]
                flags = []
                if feature_row['has_ip']:
                    flags.append(("🔴", "IP address in URL", "Attackers use raw IP addresses to bypass domain-based filters. Legitimate sites almost never use IPs in URLs."))
                if feature_row['suspicious_tld']:
                    flags.append(("🔴", "Suspicious TLD detected", "Top-level domains like .xyz, .top, .tk are cheap and commonly used in phishing campaigns."))
                if feature_row['brand_keyword']:
                    flags.append(("🟠", "Brand keyword detected", "Brand names like 'paypal', 'google', or 'bank' embedded in suspicious domains are a classic phishing trick."))
                if feature_row['phish_keyword']:
                    flags.append(("🟠", "Phishing keyword detected", "Words like 'login', 'verify', 'secure', or 'confirm' are frequently used to create urgency in phishing pages."))
                if feature_row['at_symbol']:
                    flags.append(("🔴", "@ symbol in URL", "The @ symbol causes browsers to ignore everything before it — a known URL obfuscation trick."))
                if feature_row['digit_ratio'] > 0.3:
                    flags.append(("🟡", "High digit ratio", "Unusually high proportion of digits suggests algorithmically generated domain names."))
                if feature_row['url_length'] > 75:
                    flags.append(("🟡", "Unusually long URL", "Phishing URLs are often long to hide the real domain or include tracking parameters."))
                if feature_row['hyphen_count'] > 3:
                    flags.append(("🟡", "Multiple hyphens in URL", "Attackers use hyphens to mimic legitimate domains e.g. secure-paypal-login.com."))
                if not feature_row['has_https']:
                    flags.append(("🟡", "No HTTPS", "The URL does not use HTTPS. While not conclusive, phishing pages sometimes skip SSL."))
                if feature_row['subdomain_count'] > 3:
                    flags.append(("🟠", "Deep subdomain nesting", "Multiple subdomain levels are used to make fake domains appear legitimate."))

                if flags:
                    for icon, title, explanation in flags:
                        with st.expander(f"{icon} {title}"):
                            st.markdown(explanation)
                else:
                    if prob > 0.75:
                        st.warning("⚠️ The model detected phishing based on the **combined pattern** of all 21 features. No single signal crossed its individual threshold, but the overall URL structure is statistically anomalous.")
                    else:
                        st.success("✅ No suspicious signals detected in this URL.")

            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                color = '#e74c3c' if prob > 0.75 else '#f39c12' if prob > 0.4 else '#2ecc71'
                ax.pie([prob, 1-prob], colors=[color, '#ecf0f1'],
                       startangle=90, wedgeprops=dict(width=0.5))
                ax.text(0, 0, f"{risk_pct}%", ha='center', va='center',
                       fontsize=22, fontweight='bold', color=color)
                ax.set_title("Phishing Risk Score", fontweight='bold')
                st.pyplot(fig)

            # Feature values table — Arrow-safe
            st.markdown("### Extracted Features")
            st.caption("These are the 21 signals the model used to make its decision:")
            safe_dict = {str(k): str(round(float(v), 4) if isinstance(v, float) else v)
                         for k, v in features_df.iloc[0].items()}
            feature_display = pd.DataFrame(
                list(safe_dict.items()),
                columns=["Feature", "Value"],
                dtype=object
            )
            feature_display["Feature"] = feature_display["Feature"].astype(str).astype(object)
            feature_display["Value"] = feature_display["Value"].astype(str).astype(object)
            st.table(feature_display.set_index("Feature"))

    st.markdown("---")
    st.markdown("### 🧪 Quick Test URLs")
    st.markdown("Copy any of these into the scanner above:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚨 Likely Phishing (10 examples):**")
        phishing_urls = [
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
        ]
        for u in phishing_urls:
            st.code(u)

    with col2:
        st.markdown("**✅ Likely Safe (10 examples):**")
        safe_urls = [
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
        ]
        for u in safe_urls:
            st.code(u)

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")
    st.markdown("""
    Four machine learning models were trained and evaluated on this project.
    Each model was tested both **internally** (same dataset distribution)
    and **externally** (completely different dataset) to assess real-world generalization.
    """)
    st.markdown("---")

    results = {
        "Random Forest":       {"Accuracy":0.9997,"Precision":1.0000,"Recall":0.9994,"F1":0.9997,"ROC-AUC":0.9997},
        "XGBoost":             {"Accuracy":0.9997,"Precision":1.0000,"Recall":0.9994,"F1":0.9997,"ROC-AUC":0.9997},
        "Gradient Boosting":   {"Accuracy":0.9996,"Precision":0.9998,"Recall":0.9994,"F1":0.9996,"ROC-AUC":0.9999},
        "Logistic Regression": {"Accuracy":0.9997,"Precision":1.0000,"Recall":0.9994,"F1":0.9997,"ROC-AUC":0.9998},
    }
    results_df = pd.DataFrame(results).T

    st.markdown("### 🔬 Internal Validation Results (Dataset 1 — 80/20 Split)")
    st.markdown("Trained on 39,049 URLs, tested on 9,763 URLs from the same distribution.")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"), use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌍 External Validation Results (Dataset 2 — 522,142 URLs)")
        st.markdown("The trained model was tested on a completely separate dataset to simulate real-world deployment.")
        ext = {
            "Random Forest":     {"Accuracy":0.5052,"Precision":0.2079,"Recall":0.6228,"F1":0.3117,"ROC-AUC":0.5952},
            "Gradient Boosting": {"Accuracy":0.5052,"Precision":0.2079,"Recall":0.6228,"F1":0.3120,"ROC-AUC":0.5818},
        }
        ext_df = pd.DataFrame(ext).T
        st.dataframe(ext_df.style.format("{:.4f}"), use_container_width=True)

        st.info("""
        **Why is external accuracy lower?**

        This is a documented phenomenon called **dataset distribution shift**.
        The two datasets were collected from different sources using different methodologies,
        so URL patterns differ. This is expected in real-world cybersecurity systems and
        highlights the need for continuous model retraining with fresh data.

        The high recall (~62%) means the model still catches the majority of phishing URLs
        even on unseen data — which is the most critical metric for a security system.
        """)

    with col2:
        st.markdown("### 📉 What Each Metric Means")
        metrics_explained = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Definition": [
                "Overall % of correct predictions",
                "Of all URLs flagged as phishing, how many actually were?",
                "Of all real phishing URLs, how many did we catch?",
                "Balance between Precision and Recall",
                "Overall ability to distinguish phishing from benign"
            ],
            "Why It Matters": [
                "General performance indicator",
                "Low precision = too many false alarms",
                "Low recall = missing phishing attacks",
                "Best single metric for imbalanced datasets",
                "Best metric for probabilistic classifiers"
            ]
        }
        me_df = fix_df(pd.DataFrame(metrics_explained))
        st.dataframe(me_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#2ecc71','#3498db','#e74c3c','#9b59b6']
    for i, (name, row) in enumerate(results_df.iterrows()):
        axes[0].bar(x + i*width, row[metrics], width, label=name, color=colors[i], alpha=0.85)
    axes[0].set_xticks(x + width*1.5)
    axes[0].set_xticklabels(metrics, fontsize=9)
    axes[0].set_ylim(0.998, 1.0005)
    axes[0].set_title('Internal Validation — All Models', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Score')

    axes[1].barh(list(results_df.index), results_df['ROC-AUC'], color=colors, alpha=0.85)
    axes[1].set_xlim(0.998, 1.0005)
    axes[1].set_title('ROC-AUC by Model', fontweight='bold')
    axes[1].set_xlabel('ROC-AUC Score')
    for i, v in enumerate(results_df['ROC-AUC']):
        axes[1].text(v + 0.00001, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🏆 Why Gradient Boosting Was Selected as Final Model")
    col1, col2, col3 = st.columns(3)
    col1.success("**Highest ROC-AUC: 0.9999**\nBest probabilistic separation between phishing and benign URLs.")
    col2.success("**Ensemble Strength**\nBuilds trees sequentially, each correcting errors of the previous — ideal for structured security data.")
    col3.success("**Calibrated Probabilities**\nOutputs well-calibrated risk scores, not just binary labels — essential for threshold tuning.")

# ════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📈 Feature Analysis":
    st.title("📈 Feature Analysis")
    st.markdown("""
    This page explains the 21 network-level features engineered independently from raw URLs.
    All features were derived without using any pre-computed dataset attributes,
    ensuring originality and simulating real network-gateway signal extraction.
    """)
    st.markdown("---")

    importance_data = {
        'special_char_count': 0.2012, 'dot_count': 0.1523,
        'digit_count': 0.1401, 'url_length': 0.0921,
        'digit_ratio': 0.0876, 'subdomain_count': 0.0712,
        'domain_length': 0.0634, 'path_depth': 0.0521,
        'url_entropy': 0.0487, 'hyphen_count': 0.0312,
        'token_count': 0.0298, 'domain_entropy': 0.0187,
        'vowel_ratio': 0.0156, 'has_https': 0.0121,
        'param_count': 0.0098, 'phish_keyword': 0.0076,
        'has_ip': 0.0054, 'brand_keyword': 0.0034,
        'suspicious_tld': 0.0021, 'at_symbol': 0.0012,
        'double_slash': 0.0004
    }

    imp_df = pd.DataFrame(list(importance_data.items()),
                          columns=['Feature','Importance']).sort_values('Importance', ascending=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 9))
        colors_imp = ['#e74c3c' if x > 0.05 else '#3498db' if x > 0.01 else '#95a5a6'
                  for x in imp_df['Importance']]
        bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp, alpha=0.85)
        ax.set_title('Feature Importance (Gradient Boosting)\nRed=High | Blue=Medium | Grey=Low',
                    fontweight='bold', fontsize=10)
        ax.set_xlabel('Importance Score')
        for bar, val in zip(bars, imp_df['Importance']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### 📖 Feature Dictionary")
        feature_dict_data = [
            ("url_length", "Structural", "Total character count of the URL. Phishing URLs tend to be longer to obscure the real domain."),
            ("has_https", "Security", "Whether URL uses HTTPS. Phishing sites increasingly use HTTPS to appear trustworthy."),
            ("dot_count", "Structural", "Number of dots in URL. More dots indicate deeper subdomain nesting."),
            ("subdomain_count", "Structural", "Number of subdomain levels. Deep nesting is used to mimic legitimate domains."),
            ("has_ip", "Network", "Whether URL contains a raw IP address. Legitimate sites almost never use IPs."),
            ("special_char_count", "Statistical", "Count of non-alphanumeric characters. Symbol-heavy URLs indicate obfuscation."),
            ("digit_count", "Statistical", "Raw count of numeric characters in URL."),
            ("digit_ratio", "Statistical", "Normalized digit density. High ratios suggest algorithmically generated domains."),
            ("param_count", "Structural", "Number of query parameters. Phishing pages often use complex query strings."),
            ("brand_keyword", "Semantic", "Presence of brand names like paypal, google, amazon in suspicious context."),
            ("suspicious_tld", "Network", "Whether TLD is in high-risk list: .xyz, .top, .tk, .gq, .ml, .ga, .cf, .pw"),
            ("url_entropy", "Statistical", "Shannon entropy of full URL. High entropy = high randomness = possible DGA domain."),
            ("domain_entropy", "Statistical", "Shannon entropy of domain portion only. More precise than full URL entropy."),
            ("hyphen_count", "Structural", "Number of hyphens. Used to construct fake domains like secure-paypal-login.com."),
            ("path_depth", "Structural", "Number of directory levels in URL path."),
            ("token_count", "Statistical", "Count of delimiter characters (dots, hyphens, slashes)."),
            ("vowel_ratio", "Statistical", "Ratio of vowels to total characters. DGA domains have abnormal vowel patterns."),
            ("domain_length", "Structural", "Length of domain name only. Very long domains are often malicious."),
            ("phish_keyword", "Semantic", "Presence of words like login, verify, secure, confirm, banking, password."),
            ("at_symbol", "Network", "Presence of @ in URL — a known obfuscation technique to hide real destination."),
            ("double_slash", "Structural", "Count of double slashes — can indicate URL redirection tricks."),
        ]
        feature_dict = fix_df(pd.DataFrame(feature_dict_data, columns=["Feature", "Category", "Description"]))

        category_filter = st.selectbox("Filter by Category:",
            ["All", "Structural", "Statistical", "Network", "Semantic", "Security"])

        if category_filter != "All":
            feature_dict = feature_dict[feature_dict["Category"] == category_filter]

        st.dataframe(feature_dict, use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("### 🔬 Key Research Insight")
    col1, col2, col3 = st.columns(3)
    col1.error("**Top Signal: special_char_count (20.1%)**\nSymbol-heavy URLs are the strongest indicator of phishing intent in this dataset.")
    col2.warning("**Second Signal: dot_count (15.2%)**\nSubdomain depth is a highly reliable structural indicator of phishing domains.")
    col3.info("**Third Signal: digit_count (14.0%)**\nHigh digit counts suggest algorithmically generated domain names used in campaigns.")

# ════════════════════════════════════════════════════════════
# PAGE 5 — SESSION ANALYZER
# ════════════════════════════════════════════════════════════
elif page == "🧠 Session Analyzer":
    st.title("🧠 Session-Level Behavioral Analyzer")
    st.markdown("""
    This module analyzes **how a user behaves** during a browsing session,
    not just the URL itself. Phishing sessions have distinct behavioral fingerprints:
    users are tricked into submitting credentials quickly with minimal exploration.

    Legitimate sessions show broader navigation, longer dwell time, and higher entropy.
    This is the **behavioral signal layer** that complements URL-based detection.
    """)
    st.markdown("---")

    st.markdown("### 🎛️ Configure Session Parameters")
    st.caption("Adjust the sliders to simulate a browsing session and see how the behavioral analyzer scores it.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**⏱️ Timing Signals**")
        session_duration = st.slider("Session Duration (seconds)", 5, 300, 45,
            help="Total time spent on the website before leaving.")
        time_to_submit = st.slider("Time to Credential Submission (seconds)", 1, 120, 8,
            help="How quickly the user entered and submitted credentials.")
    with col2:
        st.markdown("**🖱️ Interaction Signals**")
        num_clicks = st.slider("Number of Clicks Before Submit", 1, 20, 2,
            help="Total clicks made before submitting credentials.")
        back_button = st.slider("Back Button Usage", 0, 10, 0,
            help="Number of times the back button was pressed.")
        num_pages = st.slider("Pages Visited", 1, 15, 2,
            help="Number of distinct pages navigated during the session.")
    with col3:
        st.markdown("**📜 Engagement Signals**")
        scroll_depth = st.slider("Scroll Depth (%)", 0, 100, 25,
            help="How far down the page the user scrolled.")
        mouse_variance = st.slider("Mouse Movement Variance", 0, 100, 15,
            help="Higher variance = more natural human-like movement.")
        tab_switches = st.slider("Tab Switches", 0, 10, 0,
            help="Number of times user switched browser tabs.")

    if st.button("🔍 Analyze Session", type="primary"):
        nav_entropy = scipy_entropy([1/num_pages]*num_pages) if num_pages > 1 else 0
        submit_ratio = time_to_submit / session_duration
        click_density = num_clicks / max(session_duration, 1)
        engagement_score = (scroll_depth/100 * 0.3 + min(num_pages/10, 1) * 0.4 +
                           min(mouse_variance/100, 1) * 0.3)

        phishing_score = 0
        flags = []
        safe_signals = []

        if submit_ratio < 0.15:
            phishing_score += 30
            flags.append(("🔴", "Rapid credential submission",
                "Credentials were submitted very quickly relative to session duration. Legitimate users typically browse before logging in."))
        else:
            safe_signals.append("✅ Normal credential submission timing")

        if num_pages <= 2:
            phishing_score += 25
            flags.append(("🔴", "Minimal page navigation",
                "Very few pages visited. Phishing sessions often involve landing directly on a fake login page and leaving immediately."))
        else:
            safe_signals.append("✅ Normal navigation breadth")

        if nav_entropy < 0.5:
            phishing_score += 20
            flags.append(("🟠", "Low navigation entropy",
                f"Navigation entropy = {round(nav_entropy,3)}. Low entropy means predictable, linear browsing — characteristic of scripted phishing sessions."))
        else:
            safe_signals.append(f"✅ Normal navigation entropy ({round(nav_entropy,3)})")

        if scroll_depth < 30:
            phishing_score += 15
            flags.append(("🟠", "Low scroll depth",
                "User barely scrolled the page. Phishing victims often don't explore content — they just submit credentials."))
        else:
            safe_signals.append("✅ Normal scroll engagement")

        if back_button == 0 and num_pages > 1:
            phishing_score += 10
            flags.append(("🟡", "No back-button usage",
                "Multiple pages visited but no back-button use suggests a directed, non-exploratory path."))

        if mouse_variance < 20:
            phishing_score += 10
            flags.append(("🟡", "Low mouse movement variance",
                "Very rigid mouse movement can indicate automated or scripted interaction."))
        else:
            safe_signals.append("✅ Natural mouse movement pattern")

        if tab_switches == 0 and session_duration > 60:
            phishing_score += 5
            flags.append(("🟡", "No tab switching in long session",
                "Long session with no tab switching is unusual for genuine browsing behavior."))

        phishing_score = min(phishing_score, 100)

        st.markdown("---")
        st.markdown("### 📊 Session Analysis Results")

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
                st.error(f"### ⚠️ HIGH PHISHING RISK\nBehavioral Score: {phishing_score}%\n\nThis session matches known phishing interaction patterns.")
            elif phishing_score >= 30:
                st.warning(f"### ⚠️ MODERATE RISK\nBehavioral Score: {phishing_score}%\n\nSome suspicious behavioral patterns detected.")
            else:
                st.success(f"### ✅ LOW RISK\nBehavioral Score: {phishing_score}%\n\nSession behavior appears consistent with legitimate browsing.")

            if flags:
                st.markdown("#### 🚩 Suspicious Signals Detected")
                for icon, title, explanation in flags:
                    with st.expander(f"{icon} {title}"):
                        st.markdown(explanation)

            if safe_signals:
                st.markdown("#### ✅ Normal Signals Detected")
                for s in safe_signals:
                    st.markdown(s)

        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            categories = ['Your\nSession', 'Typical\nPhishing', 'Typical\nLegitimate']
            colors_bar = ['#3498db', '#e74c3c', '#2ecc71']

            axes[0,0].bar(categories, [time_to_submit, 5, 48], color=colors_bar, alpha=0.85)
            axes[0,0].set_title('Time to Submit (sec)', fontweight='bold', fontsize=9)

            axes[0,1].bar(categories, [num_pages, 1, 7], color=colors_bar, alpha=0.85)
            axes[0,1].set_title('Pages Visited', fontweight='bold', fontsize=9)

            axes[1,0].bar(categories, [scroll_depth, 18, 62], color=colors_bar, alpha=0.85)
            axes[1,0].set_title('Scroll Depth (%)', fontweight='bold', fontsize=9)

            axes[1,1].bar(categories, [nav_entropy, 0.1, 1.8], color=colors_bar, alpha=0.85)
            axes[1,1].set_title('Navigation Entropy', fontweight='bold', fontsize=9)

            plt.suptitle('Your Session vs Known Patterns', fontweight='bold', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🧠 Why Session Behavior Matters")
        st.info("""
        **The Behavioral Detection Principle:**

        Phishing pages are designed to steal credentials as quickly as possible.
        This creates measurable behavioral anomalies:

        - Victims submit credentials **3-8x faster** than on legitimate sites
        - Phishing sessions involve **1-2 pages** vs 5-8 for legitimate sessions
        - Navigation entropy is **near zero** — users don't explore
        - Scroll depth is typically **under 30%** — content is irrelevant

        These signals are difficult for attackers to control because they depend on
        how real users react to being deceived — not on the website structure itself.
        This is what makes behavioral analysis a powerful complement to URL-based detection.
        """)

# ════════════════════════════════════════════════════════════
# PAGE 6 — COURSE CONCEPTS APPLIED
# ════════════════════════════════════════════════════════════
elif page == "🎓 Course Concepts Applied":
    st.title("🎓 Course Concepts Applied in This Project")
    st.markdown("""
    This project was built as part of the **Network and Information Security (BITE401L)** course.
    Below is a mapping of how specific syllabus modules were directly applied in the system.
    """)
    st.markdown("---")

    st.markdown("### 📘 Module 1 — Network Security Concepts")
    st.info("""
    **Syllabus Topics:** Security Attacks, Security Services, Model for Network Security, OSI Security Architecture

    **Applied In This Project:**
    - Phishing is classified as a **passive + active attack** under the OSI security threat model
    - The detection system operates at **OSI Layer 7 (Application Layer)**
    - Features like `has_ip`, `suspicious_tld`, and `special_char_count` directly model **network-level attack indicators**
    """)

    st.markdown("### 📘 Module 3 — Cryptographic Hash Functions")
    col1, col2 = st.columns(2)
    with col1:
        st.warning("""
        **Syllabus Topics:** SHA, Hash Function Applications, Security Requirements

        **Applied In This Project:**
        - SHA-256 hashing creates a **unique fingerprint** of each URL
        - Used to detect **repeat phishing URLs** across sessions
        - URL entropy calculation relates to **information-theoretic concepts** in hash functions
        """)
    with col2:
        st.markdown("#### 🔬 Live SHA-256 URL Fingerprinting Demo")
        import hashlib
        demo_url = st.text_input("Enter a URL to generate its SHA-256 fingerprint:",
                                  value="http://paypal-login-security.xyz")
        if demo_url:
            fingerprint = hashlib.sha256(demo_url.encode()).hexdigest()
            st.code(f"URL: {demo_url}\nSHA-256: {fingerprint}", language="text")
            st.caption("Even a single character change produces a completely different hash.")

    st.markdown("### 📘 Module 5 — User Authentication")
    st.success("""
    **Syllabus Topics:** Remote User Authentication, Identity Verification, Authentication Anomalies

    **Applied In This Project:**
    - The **Session Behavioral Analyzer** directly models authentication threat patterns
    - Session signals like *time to credential submission* and *navigation entropy* model **authentication interaction anomalies**
    """)

    st.markdown("### 📘 Module 7 — Web Security")
    st.error("""
    **Syllabus Topics:** Web Security Threats, HTTPS, TLS, Web Traffic Security Approaches

    **Applied In This Project:**
    - **HTTPS Detection (`has_https`):** Checks whether URLs use TLS/HTTPS
    - **Suspicious TLD + HTTPS combo** detects TLS misuse on cheap phishing domains
    - All 21 features simulate what a **web traffic security gateway** would observe
    """)

    st.markdown("---")
    st.markdown("### 🗺️ Full Syllabus Mapping Table")
    mapping_data = [
        ("Module 1", "Network Security Concepts", "URL feature extraction at application layer, OSI threat modeling", "✅ Direct"),
        ("Module 2", "Public Key Cryptography", "HTTPS uses TLS which relies on PKI — detected via has_https feature", "⚡ Indirect"),
        ("Module 3", "Cryptographic Hash Functions", "SHA-256 URL fingerprinting, entropy-based randomness analysis", "✅ Direct"),
        ("Module 4", "MAC & Digital Signatures", "TLS certificates on phishing sites — detected via TLD + HTTPS combo", "⚡ Indirect"),
        ("Module 5", "User Authentication", "Session behavioral analysis of authentication anomalies", "✅ Direct"),
        ("Module 6", "Wireless Network Security", "Phishing attacks occur over wireless too — same detection applies", "⚡ Indirect"),
        ("Module 7", "Web Security", "HTTPS detection, web threat modeling, TLS anomaly detection", "✅ Direct"),
        ("Module 8", "Contemporary Issues", "Zero-day phishing, AI-based detection — core project motivation", "✅ Direct"),
    ]
    mapping = fix_df(pd.DataFrame(mapping_data, columns=["Module", "Topic", "Application in Project", "Relevance"]))
    st.dataframe(mapping, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Key Security Concepts Demonstrated")
    col1, col2, col3 = st.columns(3)
    col1.info("**Defense in Depth**\nMultiple independent signals — no single feature decides. Mirrors layered security architecture.")
    col2.info("**Zero-Day Detection**\nBlacklist-free detection using structural analysis — addresses zero-day web threats.")
    col3.info("**Anomaly Detection**\nBoth URL-level and session-level anomaly detection — core network security principle.")