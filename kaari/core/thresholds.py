"""
Kaari Thresholds & Model Calibration
=====================================
Per-model calibration values. 7 values per model.

FREE TIER:  Uses DEFAULT_CONFIG (generic threshold from paper).
PAID TIER:  Uses per-model calibration (computed from extensive research data).

The calibration quality IS the product differentiator.

Zone system (v0.95):
  GREEN:   score < 0.210  -- no alert, silent pass
  YELLOW:  0.210 <= score < 0.245  -- elevated deviation, review recommended
  RED:     score >= 0.245  -- high deviation, potential injection

Calibration standard: 20260216-001.md
Corrected to Option B (raw prompt) values: 20260220-001.md
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Zone boundaries (C2 metric, global calibration)
# ---------------------------------------------------------------------------
ZONE_GREEN_MAX = 0.210    # Below this: silent pass
ZONE_YELLOW_MAX = 0.245   # Below this: elevated (review). At/above: red.

ZONES = {
    "green": {"max": ZONE_GREEN_MAX, "label": "CLEAN", "action": "pass"},
    "yellow": {"min": ZONE_GREEN_MAX, "max": ZONE_YELLOW_MAX, "label": "ELEVATED", "action": "review"},
    "red": {"min": ZONE_YELLOW_MAX, "label": "INJECTION", "action": "alert"},
}


def classify_zone(score: float) -> str:
    """Classify a C2 score into green/yellow/red zone."""
    if score < ZONE_GREEN_MAX:
        return "green"
    elif score < ZONE_YELLOW_MAX:
        return "yellow"
    else:
        return "red"


# ---------------------------------------------------------------------------
# Generic defaults (from paper, Option B raw prompt pipeline)
# ---------------------------------------------------------------------------
# AUC values corrected to reflect Option B (raw prompt embedding).
# Previous v0.2.1 cited Option A values (0.870/0.883) which used LLM
# summarization. Option B was chosen for production (no LLM dependency)
# but has lower AUC. Honest numbers from kaari_calibration_v1.json.
#
# Threshold adjusted from Youden-optimal 0.239 to 0.245 to reduce
# false positives in production use.
DEFAULT_CONFIG = {
    "clean_dv2_mean": 0.1735,     # Global clean dv2 mean (N=1,115)
    "clean_dv2_std": 0.0847,      # Global clean dv2 std
    "clean_length_mean": 773.8,   # Clean response mean length (chars)
    "threshold_dv2": 0.309,       # Youden-optimal dv2 threshold
    "threshold_c2": 0.245,        # Adjusted C2 threshold (reduced FP)
    "threshold_paranoid": 0.22,   # Lower threshold for paranoid (more sensitive)
    "auc_dv2": 0.770,             # Option B dv2 AUC (N=2,228)
    "auc_c2": 0.822,              # Option B C2 AUC (N=2,228)
}


# ---------------------------------------------------------------------------
# Per-model calibration (populated by C2-A1 extractor)
# ---------------------------------------------------------------------------
# Format per model: 7 standardized values
#   clean_dv2_mean, clean_dv2_std, clean_length_mean,
#   threshold_dv2, threshold_c2, auc_dv2, auc_c2
#
# These values are computed from research data (N=486+ per model).
# In the paid API, these are served from the database.
# In the free tier, DEFAULT_CONFIG is used.

MODEL_CALIBRATION = {
    # Populated by running: python -m kaari.calibrate
    # placeholder structure — values extracted by C2-A1
}


def get_config(model_name: Optional[str] = None) -> dict:
    """
    Get scoring configuration.

    Args:
        model_name: Optional model identifier (e.g., "mistral-7b").
                    If None or not calibrated, returns DEFAULT_CONFIG.

    Returns:
        Config dict with threshold and calibration values.
    """
    if model_name and model_name in MODEL_CALIBRATION:
        return MODEL_CALIBRATION[model_name]
    return DEFAULT_CONFIG.copy()


def get_model_config(model_name: str) -> dict:
    """Get config for a specific model, falling back to defaults."""
    return get_config(model_name)


def is_calibrated(model_name: str) -> bool:
    """Check if a model has per-model calibration."""
    return model_name in MODEL_CALIBRATION


def list_calibrated_models() -> list:
    """List all models with per-model calibration."""
    return list(MODEL_CALIBRATION.keys())
