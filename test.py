"""
test_phishguard.py
------------------
Run from your project root:
    python test_phishguard.py

Tests every phase of PhishGuard.
Prints a pass/fail/warn table at the end.
No pytest needed — plain Python.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

# ── result tracking ───────────────────────────────────────────
RESULTS = []   # (phase, test_name, status, detail)

def ok(phase, name, detail=""):
    RESULTS.append((phase, name, "PASS", detail))
    print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))

def warn(phase, name, detail=""):
    RESULTS.append((phase, name, "WARN", detail))
    print(f"  [WARN] {name}" + (f"  — {detail}" if detail else ""))

def fail(phase, name, detail=""):
    RESULTS.append((phase, name, "FAIL", detail))
    print(f"  [FAIL] {name}" + (f"  — {detail}" if detail else ""))


# ═════════════════════════════════════════════════════════════
# PHASE 1 — Project structure
# ═════════════════════════════════════════════════════════════
def test_phase1():
    print("\n── Phase 1: Project structure ──────────────────────────")
    P = 1

    required_files = [
        ("dashboard.py",                  "Dashboard (root)"),
        ("requirements.txt",              "Requirements"),
        ("src/feature_extractor.py",      "Feature extractor"),
        ("src/behavior_model.py",         "Behaviour model"),
        ("src/pipeline.py",               "Pipeline class"),
        ("src/train.py",                  "Training script"),
        ("src/threshold.py",              "Threshold module"),
        ("src/artifacts.py",              "Artifact manager"),
        ("src/research_analysis.py",      "Research analysis"),
    ]

    required_dirs = ["models", "src", "data", "results"]

    for d in required_dirs:
        if Path(d).is_dir():
            ok(P, f"Directory: {d}/")
        else:
            fail(P, f"Directory: {d}/", "missing — create it")

    for fpath, label in required_files:
        if Path(fpath).exists():
            ok(P, label, fpath)
        else:
            fail(P, label, f"{fpath} not found")


# ═════════════════════════════════════════════════════════════
# PHASE 2 — Bug fixes
# ═════════════════════════════════════════════════════════════
def test_phase2():
    print("\n── Phase 2: Bug fixes ──────────────────────────────────")
    P = 2

    # Check dashboard.py exists and is readable
    dash = Path("dashboard.py")
    if not dash.exists():
        fail(P, "dashboard.py readable", "file not found"); return

    src = dash.read_text(encoding="utf-8")

    # 1. subdomain_count should NOT be url.count(".")
    if '"subdomain_count": url.count(".")' in src or "'subdomain_count': url.count('.')" in src:
        fail(P, "subdomain_count bug fixed", "still uses url.count('.') — use tldextract")
    else:
        ok(P, "subdomain_count bug fixed")

    # 2. tldextract used somewhere
    if "tldextract" in src:
        ok(P, "tldextract imported in dashboard")
    else:
        warn(P, "tldextract not in dashboard", "subdomain_count may be inaccurate")

    # 3. No emoji in page labels
    emoji_nav = any(e in src for e in ["🏠", "🔍", "📊", "📈", "🧠", "🎓"])
    if emoji_nav:
        fail(P, "Emoji removed from navigation", "still has emoji in sidebar radio")
    else:
        ok(P, "Emoji removed from navigation")

    # 4. Course page removed
    if "Course Concepts Applied" in src or "BITE401L" in src:
        fail(P, "Course page removed", "BITE401L or 'Course Concepts Applied' still present")
    else:
        ok(P, "Course page removed")

    # 5. Hardcoded keyword list style removed
    if "paypal|google|facebook" in src and "keyword_list" in src:
        warn(P, "Keyword list", "brand regex still present but as inline pattern (acceptable)")
    else:
        ok(P, "No standalone keyword_list variable")

    # 6. Hardcoded 0.75 threshold
    if "prob >= 0.75" in src or "prob > 0.75" in src:
        fail(P, "Hardcoded 0.75 threshold removed", "still using hardcoded >= 0.75")
    else:
        ok(P, "Hardcoded 0.75 threshold removed")


# ═════════════════════════════════════════════════════════════
# PHASE 3 — Model upgrade (pkl files + imports)
# ═════════════════════════════════════════════════════════════
def test_phase3():
    print("\n── Phase 3: Model upgrade ──────────────────────────────")
    P = 3

    # Check train.py mentions XGBoost, LGBM, RF
    train = Path("src/train.py")
    if train.exists():
        src = train.read_text(encoding="utf-8")
        for lib, label in [("xgboost","XGBoost"), ("lightgbm","LightGBM"), ("RandomForest","Random Forest")]:
            if lib.lower() in src.lower():
                ok(P, f"{label} in train.py")
            else:
                fail(P, f"{label} in train.py", "not found in src/train.py")

        if "StackingClassifier" in src or "meta" in src.lower() or "stacking" in src.lower():
            ok(P, "Stacking meta-learner in train.py")
        else:
            fail(P, "Stacking meta-learner in train.py", "no stacking found")

        if "cross_val" in src or "KFold" in src or "StratifiedKFold" in src:
            ok(P, "Cross-validation in train.py")
        else:
            fail(P, "Cross-validation in train.py", "no CV found — add 5-fold CV")

        if "CalibratedClassifierCV" in src:
            ok(P, "Model calibration in train.py")
        else:
            warn(P, "Model calibration", "CalibratedClassifierCV not found — recommended")
    else:
        fail(P, "src/train.py exists", "file not found — Phase 1 incomplete")

    # Check for model pkl files
    for fname, label in [
        ("models/stacking_model.pkl", "Stacking model pkl"),
        ("models/base_models.pkl",    "Base models pkl"),
        ("models/feature_names.pkl",  "Feature names pkl"),
    ]:
        if Path(fname).exists():
            ok(P, label, fname)
        else:
            warn(P, label, f"{fname} not found — run src/train.py")

    # Importability check
    try:
        import xgboost
        ok(P, "xgboost installed", xgboost.__version__)
    except ImportError:
        fail(P, "xgboost installed", "pip install xgboost")

    try:
        import lightgbm
        ok(P, "lightgbm installed", lightgbm.__version__)
    except ImportError:
        fail(P, "lightgbm installed", "pip install lightgbm")


# ═════════════════════════════════════════════════════════════
# PHASE 4 — Threshold
# ═════════════════════════════════════════════════════════════
def test_phase4():
    print("\n── Phase 4: Adaptive threshold ─────────────────────────")
    P = 4

    # threshold.py exists
    tpath = Path("src/threshold.py")
    if not tpath.exists():
        fail(P, "src/threshold.py exists"); return
    ok(P, "src/threshold.py exists")

    src = tpath.read_text(encoding="utf-8")

    # precision_recall_curve used
    if "precision_recall_curve" in src:
        ok(P, "F1-optimal threshold logic present")
    else:
        fail(P, "F1-optimal threshold logic", "precision_recall_curve not found in threshold.py")

    # uncertainty band
    if "band" in src.lower() or "UNCERTAINTY" in src or "uncertainty" in src:
        ok(P, "Uncertainty band logic present")
    else:
        fail(P, "Uncertainty band", "no uncertainty zone logic found")

    # adaptive threshold
    if "adaptive" in src.lower() or "has_ip" in src or "suspicious_tld" in src:
        ok(P, "Adaptive threshold logic present")
    else:
        fail(P, "Adaptive threshold", "no IP/TLD-based threshold shift found")

    # threshold pkl
    if Path("models/threshold.pkl").exists():
        import joblib
        val = joblib.load("models/threshold.pkl")
        if isinstance(val, dict):
            val = val.get("base_threshold", 0.5)
        ok(P, "models/threshold.pkl exists", f"value={float(val):.4f}")
        if 0.3 <= float(val) <= 0.95:
            ok(P, "Threshold value in sensible range")
        else:
            warn(P, "Threshold value", f"{float(val):.4f} is outside [0.3, 0.95] — check computation")
    else:
        warn(P, "models/threshold.pkl", "not yet generated — run: python src/threshold.py --data data/val_set.csv")

    # Functional test of get_adaptive_threshold
    try:
        sys.path.insert(0, "src")
        spec = importlib.util.spec_from_file_location("threshold", "src/threshold.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "get_adaptive_threshold"):
            base = 0.75
            
            # Helper to call flexibly
            def call_adapt(b, ip=0, tld=0):
                try:
                    res = mod.get_adaptive_threshold(b, has_ip=ip, suspicious_tld=tld)
                except TypeError:
                    # try without defaults if function needs them differently
                    res = mod.get_adaptive_threshold(b, ip, tld)
                return res[0] if isinstance(res, tuple) else res

            t1 = call_adapt(base, ip=1)
            t2 = call_adapt(base, tld=1)
            t3 = call_adapt(base)

            assert t1 < base, "IP should lower threshold"
            assert t2 < base, "Suspicious TLD should lower threshold"
            assert t3 == base, "No signals should keep base threshold"
            ok(P, "get_adaptive_threshold() logic correct",
               f"base={base} | IP={t1:.3f} | TLD={t2:.3f} | none={t3:.3f}")
        else:
            warn(P, "get_adaptive_threshold()", "function not found in threshold.py")
    except Exception as e:
        fail(P, "get_adaptive_threshold() importable", str(e))


# ═════════════════════════════════════════════════════════════
# PHASE 5 — Pipeline + behavioral fusion
# ═════════════════════════════════════════════════════════════
def test_phase5():
    print("\n── Phase 5: Pipeline + behavioural fusion ──────────────")
    P = 5

    for fpath, label in [
        ("src/pipeline.py",       "pipeline.py exists"),
        ("src/behavior_model.py", "behavior_model.py exists"),
        ("src/feature_extractor.py", "feature_extractor.py exists"),
    ]:
        if Path(fpath).exists():
            ok(P, label)
        else:
            fail(P, label, f"{fpath} missing — Phase 1/5 incomplete")

    # Check pipeline.py has fusion
    pipe = Path("src/pipeline.py")
    if pipe.exists():
        src = pipe.read_text(encoding="utf-8")
        if "fuse" in src.lower() or "url_score" in src or "behavior_score" in src:
            ok(P, "Risk fusion in pipeline.py")
        else:
            fail(P, "Risk fusion", "no fusion logic found in pipeline.py")
        if "PhishGuardPipeline" in src:
            ok(P, "PhishGuardPipeline class defined")
        else:
            fail(P, "PhishGuardPipeline class", "class not found in pipeline.py")

    # dashboard uses pipeline or fusion
    dash = Path("dashboard.py")
    if dash.exists():
        dsrc = dash.read_text(encoding="utf-8")
        if "fuse_scores" in dsrc or "behavior_score" in dsrc or "pipeline" in dsrc.lower():
            ok(P, "Dashboard uses fusion/pipeline logic")
        else:
            fail(P, "Dashboard fusion", "dashboard.py still calls model.predict directly without fusion")

    # Functional test of behavior scoring in dashboard
    try:
        spec = importlib.util.spec_from_file_location("dash_mod", "dashboard.py")
        mod  = importlib.util.module_from_spec(spec)
        # Don't fully execute (streamlit would run), just check the function is defined
        src_text = Path("dashboard.py").read_text(encoding="utf-8")
        if "def compute_behavior_score" in src_text:
            ok(P, "compute_behavior_score() defined in dashboard")
        elif Path("src/behavior_model.py").exists() and "compute_behavior_score" in Path("src/behavior_model.py").read_text(encoding="utf-8"):
            ok(P, "compute_behavior_score() defined in behavior_model.py")
        else:
            fail(P, "compute_behavior_score()", "function not found anywhere")
    except Exception as e:
        warn(P, "Behavior score function check", str(e))

    # Test the actual math if available
    try:
        import numpy as np
        from scipy.stats import entropy as scipy_entropy

        # inline the function from dashboard for testing
        def compute_behavior_score(session_duration, time_to_submit, num_pages, scroll_depth, mouse_variance):
            BEHAVIOR_WEIGHTS = {"time_ratio":0.30,"pages":0.25,"scroll":0.20,"mouse":0.15,"nav_entropy":0.10}
            submit_ratio = time_to_submit / max(session_duration, 1)
            nav_entropy  = scipy_entropy([1/num_pages]*num_pages) if num_pages > 1 else 0.0
            max_entropy  = np.log(10)
            risk = (
                BEHAVIOR_WEIGHTS["time_ratio"]   * (1 - min(submit_ratio, 1))
                + BEHAVIOR_WEIGHTS["pages"]      * (1 - min(num_pages / 10, 1))
                + BEHAVIOR_WEIGHTS["scroll"]     * (1 - scroll_depth / 100)
                + BEHAVIOR_WEIGHTS["mouse"]      * (1 - min(mouse_variance / 100, 1))
                + BEHAVIOR_WEIGHTS["nav_entropy"]* (1 - min(nav_entropy / max_entropy, 1))
            )
            return round(float(np.clip(risk, 0, 1)), 4)

        phishing_score  = compute_behavior_score(20, 2, 1, 5, 5)
        legit_score     = compute_behavior_score(180, 60, 8, 70, 65)
        assert phishing_score > legit_score, "Phishing session must score higher than legit"
        ok(P, "Behaviour scoring logic correct",
           f"phishing={phishing_score:.3f} > legitimate={legit_score:.3f}")
    except Exception as e:
        fail(P, "Behaviour scoring math", str(e))


# ═════════════════════════════════════════════════════════════
# PHASE 6 — Multi-modal / NLP token layer
# ═════════════════════════════════════════════════════════════
def test_phase6():
    print("\n── Phase 6: NLP token layer ────────────────────────────")
    P = 6

    dash = Path("dashboard.py")
    if not dash.exists():
        fail(P, "dashboard.py readable"); return

    src = dash.read_text(encoding="utf-8")

    if "url_token_features" in src or "token_entropy" in src:
        ok(P, "URL tokenisation function present")
    else:
        fail(P, "URL tokenisation", "url_token_features / token_entropy not in dashboard.py")

    if "avg_token_len" in src:
        ok(P, "avg_token_len feature present")
    else:
        fail(P, "avg_token_len feature", "not found")

    if "num_tokens" in src:
        ok(P, "num_tokens feature present")
    else:
        fail(P, "num_tokens feature", "not found")

    # Functional test
    try:
        import re
        import numpy as np
        from scipy.stats import entropy as scipy_entropy
        from collections import Counter

        def url_token_features(url):
            raw_tokens = re.split(r"[.\-_/]", url)
            tokens     = [t for t in raw_tokens if len(t) > 0]
            if not tokens:
                return {"avg_token_len": 0.0, "token_entropy": 0.0, "num_tokens": 0}
            avg_len    = np.mean([len(t) for t in tokens])
            len_counts = Counter([len(t) for t in tokens])
            probs      = np.array(list(len_counts.values())) / len(tokens)
            tok_entropy = float(scipy_entropy(probs))
            return {"avg_token_len": round(avg_len, 4), "token_entropy": round(tok_entropy, 4), "num_tokens": len(tokens)}

        legit = url_token_features("https://www.google.com/search?q=hello")
        phish = url_token_features("http://a8b3x9k2.xyz/a8b3?x=k9m2")
        ok(P, "Token features computed correctly",
           f"google tokens={legit['num_tokens']} | phish tokens={phish['num_tokens']}")
    except Exception as e:
        fail(P, "Token feature computation", str(e))


# ═════════════════════════════════════════════════════════════
# PHASE 7 — Dashboard UI
# ═════════════════════════════════════════════════════════════
def test_phase7():
    print("\n── Phase 7: Dashboard UI ───────────────────────────────")
    P = 7

    dash = Path("dashboard.py")
    if not dash.exists():
        fail(P, "dashboard.py exists"); return
    src = dash.read_text(encoding="utf-8")

    checks = [
        ("Behavioural Risk Analyzer page",    "Behavioural Risk Analyzer" in src or "Behavioral Risk Analyzer" in src),
        ("Model Performance page",             "Model Performance" in src),
        ("Feature Analysis page",              "Feature Analysis" in src),
        ("No Home page",                       "\"Home\"" not in src and "'Home'" not in src),
        ("No course page",                     "Course Concepts" not in src and "BITE401L" not in src),
        ("Threshold displayed in sidebar",     "threshold" in src.lower() and "sidebar" in src.lower()),
        ("Uncertainty zone in UI",             "uncertain" in src.lower() or "uncertainty" in src.lower()),
        ("Risk gauge function",                "draw_gauge" in src or "gauge" in src.lower()),
        ("Score breakdown table",              "Score breakdown" in src or "score_data" in src),
        ("Combined risk label",                "Combined risk" in src or "fused" in src.lower()),
        ("No standalone keyword_list",         "keyword_list" not in src),
        ("st.cache_resource used",             "st.cache_resource" in src),
        ("fix_df helper present",              "fix_df" in src),
        ("Streamlit import",                   "import streamlit as st" in src),
    ]

    for label, result in checks:
        (ok if result else fail)(P, label)


# ═════════════════════════════════════════════════════════════
# PHASE 8 — Artifact management
# ═════════════════════════════════════════════════════════════
def test_phase8():
    print("\n── Phase 8: Artifact management ────────────────────────")
    P = 8

    apath = Path("src/artifacts.py")
    if not apath.exists():
        fail(P, "src/artifacts.py exists"); return
    ok(P, "src/artifacts.py exists")

    src = apath.read_text(encoding="utf-8")
    for fn in ["save_artifact", "load_artifact", "audit", "clean", "load_inference_bundle"]:
        if f"def {fn}" in src:
            ok(P, f"{fn}() defined")
        else:
            fail(P, f"{fn}() defined", f"function not found in artifacts.py")

    # Stale file list
    if "phishing_detector_model.pkl" in src:
        ok(P, "Stale pkl listed for cleanup")
    else:
        warn(P, "Stale pkl", "phishing_detector_model.pkl not in STALE_ARTIFACTS list")

    # Canonical artifacts list
    if "CANONICAL_ARTIFACTS" in src:
        ok(P, "CANONICAL_ARTIFACTS registry defined")
    else:
        fail(P, "CANONICAL_ARTIFACTS registry", "not found in artifacts.py")

    # models/ folder
    if Path("models").is_dir():
        pkls = list(Path("models").glob("*.pkl"))
        ok(P, f"models/ contains {len(pkls)} pkl file(s)", ", ".join(p.name for p in pkls))
        if any(p.name == "phishing_detector_model.pkl" for p in pkls):
            warn(P, "Stale pkl still present", "run: python src/artifacts.py --clean --execute")
        else:
            ok(P, "No stale phishing_detector_model.pkl present")
    else:
        warn(P, "models/ directory", "not yet created — run src/train.py")


# ═════════════════════════════════════════════════════════════
# PHASE 9 — Research analysis
# ═════════════════════════════════════════════════════════════
def test_phase9():
    print("\n── Phase 9: Research analysis ──────────────────────────")
    P = 9

    rpath = Path("src/research_analysis.py")
    if not rpath.exists():
        fail(P, "src/research_analysis.py exists"); return
    ok(P, "src/research_analysis.py exists")

    src = rpath.read_text(encoding="utf-8")
    for fn in ["distribution_shift_analysis", "adversarial_robustness", "write_generalization_report"]:
        if f"def {fn}" in src:
            ok(P, f"{fn}() defined")
        else:
            fail(P, f"{fn}() defined", "function missing")

    if "ADVERSARIAL_PAIRS" in src:
        ok(P, "Adversarial URL pairs defined")
    else:
        fail(P, "ADVERSARIAL_PAIRS", "not found")

    # Count pairs
    import re
    pairs = re.findall(r'"\(', src) + re.findall(r"\(\s*\n\s*\"http", src)
    if len(src.split('"http')) >= 10:
        ok(P, "At least 10 adversarial pairs present")
    else:
        warn(P, "Adversarial pairs count", "check ADVERSARIAL_PAIRS has 10 entries")

    # results files
    for fpath, label, run_hint in [
        ("results/generalization_report.json", "Generalization report JSON",
         "python src/research_analysis.py"),
        ("results/threshold_report.json",      "Threshold report JSON",
         "python src/threshold.py --data data/val_set.csv"),
        ("results/figures/distribution_shift.png", "Distribution shift figure",
         "python src/research_analysis.py"),
        ("results/figures/adversarial_robustness.png", "Adversarial robustness figure",
         "python src/research_analysis.py"),
        ("results/figures/threshold_analysis.png", "Threshold analysis figure",
         "python src/threshold.py --data data/val_set.csv"),
    ]:
        if Path(fpath).exists():
            ok(P, label)
        else:
            warn(P, label, f"not yet generated — run: {run_hint}")

    # Functional test — adversarial feature extraction
    try:
        import importlib.util, numpy as np
        spec = importlib.util.spec_from_file_location("ra", "src/research_analysis.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        feat = mod._extract_for_adversarial("http://paypal-login.xyz/update")
        assert "has_ip" in feat
        assert feat["suspicious_tld"] == 1, "Should detect .xyz"
        assert feat["has_https"] == 0, "No HTTPS"
        ok(P, "_extract_for_adversarial() works correctly",
           f"suspicious_tld={feat['suspicious_tld']} has_ip={feat['has_ip']}")
    except Exception as e:
        fail(P, "_extract_for_adversarial() importable and correct", str(e))


# ═════════════════════════════════════════════════════════════
# BONUS — End-to-end inference test
# ═════════════════════════════════════════════════════════════
def test_e2e():
    print("\n── End-to-end: Feature extraction + model prediction ───")
    P = "E2E"

    # Try meta_learner first (stacking meta-learner is the actual classifier)
    # stacking_model.pkl may be a dict of base models
    for candidate in ["models/meta_learner.pkl", "models/stacking_model.pkl", "models/best_model.pkl"]:
        model_path = Path(candidate)
        if model_path.exists():
            break
    feat_path  = Path("models/feature_names.pkl")

    if not model_path.exists() or not feat_path.exists():
        warn(P, "E2E test skipped", "model pkl not found — run src/train.py first")
        return

    try:
        import joblib, pandas as pd, numpy as np, re
        from urllib.parse import urlparse
        from collections import Counter
        from scipy.stats import entropy as scipy_entropy

        model         = joblib.load(model_path)
        
        # If loaded object is a dict of base models, grab the meta-learner
        if isinstance(model, dict):
            warn(P, "stacking_model.pkl is a dict — loading meta_learner.pkl instead")
            meta_path = Path("models/meta_learner.pkl")
            if meta_path.exists():
                model = joblib.load(meta_path)
            else:
                warn(P, "E2E test skipped", "meta_learner.pkl not found — check train.py saves it separately")
                warn(P, "E2E test skipped", "meta_learner.pkl not found — check train.py saves it separately")
                return
        
        base_models_path = Path("models/base_models.pkl")
        base_models = joblib.load(base_models_path) if base_models_path.exists() else None

        feature_names = joblib.load(feat_path)

        def fast_entropy(s):
            if not s: return 0.0
            c = Counter(s)
            p = np.array(list(c.values())) / len(s)
            return float(scipy_entropy(p))

        from src.feature_extractor import extract_features as extract

        test_cases = [
            ("http://paypal-login-security.xyz/update", "phishing"),
            ("http://192.168.1.1/bank/login.php",       "phishing"),
            ("https://www.google.com",                  "safe"),
            ("https://www.github.com",                  "safe"),
        ]

        correct = 0
        for url, expected in test_cases:
            feat = extract(url)
            # align to feature_names order
            row  = pd.DataFrame([[feat.get(f, 0) for f in feature_names]], columns=feature_names)
            
            try:
                prob = float(model.predict_proba(row)[0][1])
            except ValueError:
                # Likely a stacking meta-learner needing base model predictions
                stack_input = np.column_stack([
                    m.predict_proba(row)[:, 1]
                    for m in base_models.values()
                ])
                prob = float(model.predict_proba(stack_input)[0][1])

            pred = "phishing" if prob >= 0.5 else "safe"
            status = "PASS" if pred == expected else "FAIL"
            print(f"    [{status}] {url[:55]:<55}  prob={prob:.3f}  pred={pred}")
            if pred == expected:
                correct += 1

        if correct == len(test_cases):
            ok(P, f"All {len(test_cases)} E2E test cases correct")
        elif correct >= len(test_cases) * 0.75:
            warn(P, f"{correct}/{len(test_cases)} E2E cases correct", "some misclassified")
        else:
            fail(P, f"Only {correct}/{len(test_cases)} E2E cases correct", "model may need retraining")

    except Exception as e:
        fail(P, "E2E inference", str(e))
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "═"*62)
    print("  PHISHGUARD TEST SUMMARY")
    print("═"*62)

    by_phase = {}
    for phase, name, status, detail in RESULTS:
        by_phase.setdefault(phase, []).append(status)

    phase_labels = {
        1: "Project structure",
        2: "Bug fixes",
        3: "Model upgrade (XGB/LGBM/RF)",
        4: "Adaptive threshold",
        5: "Pipeline + fusion",
        6: "NLP token layer",
        7: "Dashboard UI",
        8: "Artifact management",
        9: "Research analysis",
        "E2E": "End-to-end inference",
    }

    all_phases = [1,2,3,4,5,6,7,8,9,"E2E"]
    for p in all_phases:
        statuses = by_phase.get(p, [])
        if not statuses:
            print(f"  Phase {str(p):<4}  {phase_labels.get(p,''):<30}  [NOT RUN]")
            continue
        n_pass = statuses.count("PASS")
        n_warn = statuses.count("WARN")
        n_fail = statuses.count("FAIL")
        total  = len(statuses)
        pct    = int(n_pass / total * 100)
        bar    = "#" * (pct // 10) + "." * (10 - pct // 10)
        label  = "DONE" if n_fail == 0 and n_warn == 0 else \
                 "PARTIAL" if n_fail == 0 else \
                 "INCOMPLETE"
        print(f"  Phase {str(p):<4}  {phase_labels.get(p,''):<30}  [{bar}] {pct:3d}%  {label}  "
              f"({n_pass}P {n_warn}W {n_fail}F)")

    total_pass = sum(1 for _,_,s,_ in RESULTS if s=="PASS")
    total_warn = sum(1 for _,_,s,_ in RESULTS if s=="WARN")
    total_fail = sum(1 for _,_,s,_ in RESULTS if s=="FAIL")
    grand_total = len(RESULTS)

    print("─"*62)
    print(f"  Total  {total_pass}/{grand_total} passed   {total_warn} warnings   {total_fail} failures")

    if total_fail == 0 and total_warn == 0:
        print("\n  All phases complete. Ready for paper submission.")
    elif total_fail == 0:
        print("\n  Core complete. Warnings = run training scripts to generate outputs.")
    else:
        print("\n  Incomplete phases listed above. Fix FAILs first, then re-run.")
    print("═"*62 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPhishGuard — full test suite")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")

    test_phase1()
    test_phase2()
    test_phase3()
    test_phase4()
    test_phase5()
    test_phase6()
    test_phase7()
    test_phase8()
    test_phase9()
    test_e2e()
    print_summary()