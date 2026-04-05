"""
Kaari v0.95 — Intent Vectoring for Prompt Injection Detection
=============================================================
Black-box prompt injection detection via semantic deviation measurement.

Quick start:
    import kaari

    # Initialize with local Ollama (free)
    k = kaari.Kaari()

    # Score a prompt-response pair
    result = k.score("What is 2+2?", "The answer is 4.")
    print(result.zone)      # "green" — clean
    print(result.injected)  # False
    print(result.risk)      # Low number

    # Configure what happens on RED zone
    k = kaari.Kaari(on_red="raise")  # Raises exception on injection

    # Use as decorator
    @k.guard
    def my_llm_call(prompt):
        return call_my_model(prompt)

Zone system:
    GREEN:   score < 0.210  — silent pass
    YELLOW:  0.210-0.245    — elevated, review recommended
    RED:     score >= 0.245  — potential injection

References:
    Lertola, T.S. (2026). "Intent Vectoring: Black-Box Prompt Injection
    Detection via Semantic Deviation Measurement."
"""

__version__ = "0.95.0"

from kaari.client import Kaari, InjectionDetected
from kaari.core.scoring import KaariError, KaariInputError

__all__ = ["Kaari", "InjectionDetected", "KaariError", "KaariInputError", "__version__"]
