"""
Tests for Kaari core scoring engine.
Run: pytest kaari/tests/ -v
"""

import math
import numpy as np
import pytest

from kaari.core.scoring import (
    cosine_similarity,
    calculate_delta,
    calculate_c2,
    score,
    ScoringResult,
)
from kaari.core.thresholds import DEFAULT_CONFIG, get_config
from kaari.core.families import detect_family, FamilyResult


# -----------------------------------------------------------------------
# cosine_similarity
# -----------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_numpy_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = cosine_similarity(a, b)
        assert 0.0 < result < 1.0

    def test_high_dimensional(self):
        """768-dim like nomic-embed-text."""
        rng = np.random.RandomState(42)
        a = rng.randn(768)
        b = rng.randn(768)
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


# -----------------------------------------------------------------------
# calculate_delta
# -----------------------------------------------------------------------

class TestCalculateDelta:
    def test_identical_is_zero(self):
        v = [1.0, 2.0, 3.0]
        assert calculate_delta(v, v) == pytest.approx(0.0)

    def test_orthogonal_is_one(self):
        assert calculate_delta([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)

    def test_delta_range(self):
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randn(768)
            b = rng.randn(768)
            d = calculate_delta(a, b)
            assert 0.0 <= d <= 2.0


# -----------------------------------------------------------------------
# calculate_c2
# -----------------------------------------------------------------------

class TestCalculateC2:
    def test_average_length_returns_dv2(self):
        """When response length equals clean mean, C2 = Δv2."""
        assert calculate_c2(0.3, 500, 500) == pytest.approx(0.3)

    def test_longer_amplifies(self):
        """Longer response should amplify the score."""
        c2 = calculate_c2(0.3, 1000, 500)
        assert c2 > 0.3

    def test_shorter_dampens(self):
        """Shorter response should dampen the score."""
        c2 = calculate_c2(0.3, 250, 500)
        assert c2 < 0.3

    def test_zero_length_returns_dv2(self):
        assert calculate_c2(0.3, 0, 500) == 0.3

    def test_zero_mean_returns_dv2(self):
        assert calculate_c2(0.3, 500, 0) == 0.3

    def test_coefficient_is_half(self):
        """Verify the 0.5 coefficient."""
        dv2 = 0.3
        resp_len = 1000
        mean_len = 500
        expected = dv2 * (1 + 0.5 * math.log(resp_len / mean_len))
        assert calculate_c2(dv2, resp_len, mean_len) == pytest.approx(expected)


# -----------------------------------------------------------------------
# score()
# -----------------------------------------------------------------------

class TestScore:
    def setup_method(self):
        self.config = DEFAULT_CONFIG.copy()
        # Similar vectors (clean)
        rng = np.random.RandomState(42)
        base = rng.randn(768)
        self.clean_prompt = base
        self.clean_response = base + rng.randn(768) * 0.1  # Small perturbation
        # Dissimilar vectors (dirty)
        self.dirty_response = rng.randn(768)  # Completely different

    def test_clean_not_injected(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.clean_response,
            response_length=500,
            config=self.config,
            tier="fast",
        )
        assert isinstance(result, ScoringResult)
        assert result.injected is False
        assert result.delta_v2 < 0.2
        assert result.tier == "fast"

    def test_dirty_is_injected(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.dirty_response,
            response_length=1200,
            config=self.config,
            tier="standard",
        )
        assert result.injected is True
        assert result.delta_v2 > 0.5
        assert result.c2 is not None
        assert result.tier == "standard"

    def test_fast_tier_no_c2(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.clean_response,
            response_length=500,
            config=self.config,
            tier="fast",
        )
        assert result.c2 is None
        assert result.delta_v1 is None

    def test_standard_tier_has_c2(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.clean_response,
            response_length=500,
            config=self.config,
            tier="standard",
        )
        assert result.c2 is not None

    def test_risk_0_to_100(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.dirty_response,
            response_length=500,
            config=self.config,
        )
        assert 0 <= result.risk <= 100

    def test_confidence_0_to_1(self):
        result = score(
            prompt_embedding=self.clean_prompt,
            response_embedding=self.dirty_response,
            response_length=500,
            config=self.config,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            score(
                prompt_embedding=self.clean_prompt,
                response_embedding=self.clean_response,
                response_length=500,
                config=self.config,
                tier="nonexistent",
            )


# -----------------------------------------------------------------------
# Family detection
# -----------------------------------------------------------------------

class TestFamilyDetection:
    def test_nasdaq_family(self):
        response = "You should invest in NASDAQ stocks and buy shares in the market."
        result = detect_family(response)
        assert result.family == "nasdaq"
        assert result.confidence == "high"

    def test_code_family(self):
        response = "Here is a Python script:\nimport requests\ndef scrape():\n  print('done')"
        result = detect_family(response)
        assert result.family == "code"
        assert result.confidence == "high"

    def test_persona_family(self):
        response = "Welcome! I'm Marcus, your luxury travel concierge in Monaco."
        result = detect_family(response)
        assert result.family == "persona"
        assert result.confidence == "high"

    def test_clean_no_family(self):
        response = "The sum of 2 and 2 is 4."
        result = detect_family(response)
        assert result.family is None
        assert result.confidence == "low"

    def test_returns_family_result(self):
        result = detect_family("some text")
        assert isinstance(result, FamilyResult)


# -----------------------------------------------------------------------
# Thresholds / Config
# -----------------------------------------------------------------------

class TestThresholds:
    def test_default_config_has_all_keys(self):
        required = [
            "clean_dv2_mean", "clean_dv2_std", "clean_length_mean",
            "threshold_dv2", "threshold_c2", "auc_dv2", "auc_c2",
        ]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_get_config_returns_copy(self):
        c1 = get_config()
        c2 = get_config()
        c1["threshold_dv2"] = 999
        assert c2["threshold_dv2"] != 999

    def test_unknown_model_returns_default(self):
        config = get_config("nonexistent-model-xyz")
        assert config == DEFAULT_CONFIG
