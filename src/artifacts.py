CANONICAL_ARTIFACTS = {
    "stacking_model.pkl" : "Meta-learner stacking model",
    "base_models.pkl"    : "Dict of base models",
    "feature_names.pkl"  : "Ordered feature name list",
    "threshold.pkl"      : "Learned F1-optimal threshold",
    "meta_learner.pkl"   : "Logistic regression meta-learner",
    "best_model.pkl"     : "Backward-compatible model",
}

STALE_ARTIFACTS = ["phishing_detector_model.pkl"]


def save_artifact(obj, name: str, description: str = ""):
    import joblib
    from pathlib import Path
    path = Path("models") / name
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"[artifacts] saved {path}")
    return path


def load_artifact(name: str, fallback=None):
    import joblib
    from pathlib import Path
    path = Path("models") / name
    if not path.exists():
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def audit():
    from pathlib import Path
    print("\n── Artifact audit ──────────────────")
    for p in sorted(Path("models").glob("*.pkl")):
        status = "OK" if p.name in CANONICAL_ARTIFACTS else \
                 "STALE" if p.name in STALE_ARTIFACTS else "UNKNOWN"
        print(f"  [{status}] {p.name}  ({p.stat().st_size/1024:.1f} KB)")
    print()


def clean(dry_run: bool = True):
    from pathlib import Path
    for name in STALE_ARTIFACTS:
        path = Path("models") / name
        if path.exists():
            if dry_run:
                print(f"  [DRY RUN] would delete: {path}")
            else:
                path.unlink()
                print(f"  [DELETED] {path}")


def load_inference_bundle() -> dict:
    import joblib
    from pathlib import Path

    for candidate in ["stacking_model.pkl", "meta_learner.pkl", "best_model.pkl"]:
        p = Path("models") / candidate
        if p.exists():
            obj = joblib.load(p)
            if not isinstance(obj, dict):
                model = obj
                break
    else:
        raise FileNotFoundError("No usable model found in models/")

    feature_names = load_artifact("feature_names.pkl")

    try:
        thresh_val = load_artifact("threshold.pkl")
        if isinstance(thresh_val, dict):
            threshold = float(thresh_val.get("base_threshold", 0.75))
        else:
            threshold = float(thresh_val)
        threshold_source = "learned"
    except FileNotFoundError:
        threshold = 0.75
        threshold_source = "default"

    return {
        "model"            : model,
        "feature_names"    : feature_names,
        "threshold"        : threshold,
        "threshold_source" : threshold_source,
    }
