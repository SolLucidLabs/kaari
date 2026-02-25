"""Kaari Core — Intent Vectoring scoring engine."""
from kaari.core.scoring import score, calculate_delta, calculate_c2, cosine_similarity
from kaari.core.thresholds import get_config, get_model_config, DEFAULT_CONFIG
from kaari.core.families import detect_family

__all__ = [
    "score",
    "calculate_delta",
    "calculate_c2",
    "cosine_similarity",
    "get_config",
    "get_model_config",
    "DEFAULT_CONFIG",
    "detect_family",
]
