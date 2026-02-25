"""
OpenAI Embedding Provider
=========================
Server-side embeddings via OpenAI API.
Model: text-embedding-3-small (1536-dim, $0.02/1M tokens).

Used by: Pipeline 2 (Paid API), Pipeline 4 (Chrome paid).
Requires: pip install openai
"""

import numpy as np

from kaari.embeddings.base import EmbeddingProvider, EmbeddingError


class OpenAIEmbedding(EmbeddingProvider):
    """Embed text via OpenAI Embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install openai"
            )

        self._model = model
        self._client = openai.OpenAI(api_key=api_key)
        self._dim = 1536 if "small" in model else 3072

    def embed(self, text: str) -> np.ndarray:
        """Embed text via OpenAI API."""
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            vec = np.array(response.data[0].embedding, dtype=np.float64)
            return vec
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai/{self._model}"
