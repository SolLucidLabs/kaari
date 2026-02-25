"""
Kaari Thresholds & Model Calibration
=====================================
Per-model calibration values. 7 values per model.

FREE TIER:  Uses DEFAULT_CONFIG (generic threshold from paper).
PAID TIER:  Uses per-model calibration (secret, computed from research data).

The calibration quality IS the product differentiator.

Calibration standard: 20260216-001.md
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Generic defaults (from paper, public)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "clean_dv2_mean": 0.18,       # Approximate cross-model clean Δv2 mean
    "clean_dv2_std": 0.08,        # Approximate cross-model clean Δv2 std
    "clean_length_mean": 774.0,   # Clean response mean length (chars)
    "threshold_dv2": 0.291,       # Youden-optimal from paper (N=1,944)
    "threshold_c2": 0.24,         # Approximate C2 threshold
    "threshold_paranoid": 0.22,   # Lower threshold for paranoid (more sensitive)
    "auc_dv2": 0.870,             # Published AUC
    "auc_c2": 0.883,              # Published C2 AUC
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
