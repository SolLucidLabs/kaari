"""
Kaari Scoring Engine
====================
The core product. ~50 lines. numpy only.

Measures semantic deviation between user prompt and model response
using cosine distance in embedding space.

Metrics:
  Δv2 (fast):     1 - cos_sim(embed(prompt), embed(response))
  C2 (standard):  Δv2 × (1 + 0.5 × log(response_length / μ_clean_length))
  Paranoid:       C2 weighted with Δv1 (response intent embedding)

References:
  Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection
  Detection via Semantic Deviation Measurement." arXiv preprint.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ScoringResult:
    """Result of scoring a prompt-response pair."""
    score: float           # Primary score (C2 for standard, dv2 for fast)
    risk: int              # 0-100 risk level
    injected: bool         # Above threshold?
    family: Optional[str]  # Detected injection family (nasdaq/code/persona/None)
    confidence: float      # How far above/below threshold (0-1 clamped)
    delta_v2: float        # Raw Δv2 value
    delta_v1: Optional[float]  # Δv1 if paranoid tier
    c2: Optional[float]    # C2 value if standard/paranoid tier
    tier: str              # Which tier was used


def cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def calculate_delta(v1, v2) -> float:
    """Cosine distance: 1 - cosine_similarity. Higher = more deviation."""
    return 1.0 - cosine_similarity(v1, v2)


def calculate_c2(dv2: float, response_length: int, clean_mean_length: float) -> float:
    """
    Length-normalized delta. Amplifies deviation for verbose responses.
    C2 = Δv2 × (1 + 0.5 × log(response_length / μ_clean_length))

    The 0.5 coefficient is validated as optimal (20260220-003).
    """
    if clean_mean_length <= 0 or response_length <= 0:
        return dv2
    return dv2 * (1.0 + 0.5 * math.log(response_length / clean_mean_length))


def score(
    prompt_embedding,
    response_embedding,
    response_length: int,
    config: dict,
    response_intent_embedding=None,
    tier: str = "standard",
) -> ScoringResult:
    """
    Score a prompt-response pair for injection.

    Args:
        prompt_embedding:          Embedded user prompt (raw, no LLM summary)
        response_embedding:        Embedded model response
        response_length:           Character length of response text
        config:                    Model config dict (from thresholds.py)
        response_intent_embedding: Embedded response intent summary (paranoid only)
        tier:                      "fast", "standard", or "paranoid"

    Returns:
        ScoringResult with score, risk level, injection flag, and metadata.
    """
    # Δv2: prompt vs raw response (the core signal)
    dv2 = calculate_delta(prompt_embedding, response_embedding)

    # Tier-specific scoring
    if tier == "fast":
        primary_score = dv2
        threshold = config["threshold_dv2"]
        c2_val = None
        dv1_val = None

    elif tier == "standard":
        c2_val = calculate_c2(dv2, response_length, config["clean_length_mean"])
        primary_score = c2_val
        threshold = config["threshold_c2"]
        dv1_val = None

    elif tier == "paranoid":
        c2_val = calculate_c2(dv2, response_length, config["clean_length_mean"])
        if response_intent_embedding is not None:
            dv1_val = calculate_delta(prompt_embedding, response_intent_embedding)
            # Paranoid: weighted combination of C2 and Δv1
            primary_score = 0.7 * c2_val + 0.3 * dv1_val
        else:
            dv1_val = None
            primary_score = c2_val
        threshold = config.get("threshold_paranoid", config["threshold_c2"])

    else:
        raise ValueError(f"Unknown tier: {tier}. Use 'fast', 'standard', or 'paranoid'.")

    # Injection decision
    injected = primary_score >= threshold

    # Confidence: how far from threshold (clamped 0-1)
    if threshold > 0:
        confidence = min(1.0, max(0.0, abs(primary_score - threshold) / threshold))
    else:
        confidence = 1.0 if injected else 0.0

    # Risk: 0-100 scale
    risk = min(100, max(0, int(100 * primary_score / max(threshold * 2, 0.001))))

    return ScoringResult(
        score=primary_score,
        risk=risk,
        injected=injected,
        family=None,  # Filled by families.py in the pipeline
        confidence=confidence,
        delta_v2=dv2,
        delta_v1=dv1_val,
        c2=c2_val,
        tier=tier,
    )
