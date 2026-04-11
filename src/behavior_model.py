# src/behavior_model.py
# Phase 5 — Behavioral risk scorer with documented weights

from scipy.stats import entropy as scipy_entropy


# Documented weights — these go in the paper as Table N
BEHAVIOR_WEIGHTS = {
    "submit_speed":       0.30,   # strongest signal — fastest credential theft indicator
    "page_navigation":    0.25,   # minimal browsing = directed phishing session
    "navigation_entropy": 0.20,   # low entropy = non-exploratory, scripted path
    "scroll_depth":       0.15,   # low scroll = user not reading content
    "mouse_variance":     0.10,   # rigid movement = possible automation
}


def compute_behavior_score(
    session_duration: float,
    time_to_submit: float,
    num_pages: int,
    scroll_depth: float,
    mouse_variance: float,
    back_button: int = 0,
    tab_switches: int = 0,
) -> dict:
    """
    Compute a normalized behavioral risk score in [0, 1].

    Parameters match the Behavioral Risk Analyzer sliders.
    Returns a dict with score, component scores, and flags.
    """

    # ── Component 1: Submit speed ratio ──────────────────
    # Lower ratio = faster submission = higher risk
    submit_ratio  = time_to_submit / max(session_duration, 1)
    speed_risk    = 1.0 - min(submit_ratio / 0.5, 1.0)
    # submit_ratio < 0.15 → near max risk; > 0.5 → near zero risk

    # ── Component 2: Page navigation risk ─────────────────
    # Fewer pages = higher risk; normalised against 10 pages as "normal"
    nav_risk = 1.0 - min((num_pages - 1) / 9.0, 1.0)

    # ── Component 3: Navigation entropy ───────────────────
    nav_entropy  = scipy_entropy([1 / num_pages] * num_pages) if num_pages > 1 else 0
    max_entropy  = scipy_entropy([1 / 10] * 10)               # entropy at 10 pages
    entropy_risk = 1.0 - min(nav_entropy / max(max_entropy, 1e-9), 1.0)

    # ── Component 4: Scroll depth risk ────────────────────
    scroll_risk = 1.0 - (scroll_depth / 100.0)

    # ── Component 5: Mouse variance risk ──────────────────
    mouse_risk = 1.0 - min(mouse_variance / 100.0, 1.0)

    # ── Weighted fusion ───────────────────────────────────
    behavior_score = (
        BEHAVIOR_WEIGHTS["submit_speed"]       * speed_risk   +
        BEHAVIOR_WEIGHTS["page_navigation"]    * nav_risk     +
        BEHAVIOR_WEIGHTS["navigation_entropy"] * entropy_risk +
        BEHAVIOR_WEIGHTS["scroll_depth"]       * scroll_risk  +
        BEHAVIOR_WEIGHTS["mouse_variance"]     * mouse_risk
    )
    behavior_score = float(min(max(behavior_score, 0.0), 1.0))

    # ── Flags for dashboard display ───────────────────────
    flags = []
    if submit_ratio < 0.15:
        flags.append(("High", "Rapid credential submission",
            "Credentials submitted very quickly relative to session duration."))
    if num_pages <= 2:
        flags.append(("High", "Minimal page navigation",
            "Very few pages visited. Phishing sessions involve minimal browsing."))
    if nav_entropy < 0.5:
        flags.append(("Medium", "Low navigation entropy",
            f"Navigation entropy = {round(nav_entropy, 3)}. Predictable, linear path."))
    if scroll_depth < 30:
        flags.append(("Medium", "Low scroll depth",
            "User barely scrolled. Phishing victims rarely read page content."))
    if mouse_variance < 20:
        flags.append(("Low", "Low mouse movement variance",
            "Rigid movement may indicate automated or scripted interaction."))
    if back_button == 0 and num_pages > 1:
        flags.append(("Low", "No back-button usage",
            "Multiple pages with no back navigation suggests a directed path."))
    if tab_switches == 0 and session_duration > 60:
        flags.append(("Low", "No tab switching in long session",
            "Long session with no tab switching is unusual for genuine browsing."))

    return {
        "behavior_score":   behavior_score,
        "nav_entropy":      round(nav_entropy, 4),
        "submit_ratio":     round(submit_ratio, 4),
        "speed_risk":       round(speed_risk, 4),
        "nav_risk":         round(nav_risk, 4),
        "entropy_risk":     round(entropy_risk, 4),
        "scroll_risk":      round(scroll_risk, 4),
        "mouse_risk":       round(mouse_risk, 4),
        "flags":            flags,
    }