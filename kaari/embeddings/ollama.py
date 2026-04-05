"""
Ollama Embedding Provider
=========================
Local embeddings via Ollama. Free, private, no API key needed.
Default model: nomic-embed-text (768-dim, same as research).

Used by: Pipeline 1 (GitHub free), local development.
"""

import numpy as np
import requests

from kaari.embeddings.base import EmbeddingProvider, EmbeddingError


class OllamaEmbedding(EmbeddingProvider):
    """Embed text via local Ollama instance."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: tuple = (5, 90),
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/api/embeddings"
        self._timeout = timeout
        self._dimension = None  # Detected on first call

    def embed(self, text: str) -> np.ndarray:
        """Embed text via Ollama REST API."""
        try:
            response = requests.post(
                self._endpoint,
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
            )
        except requests.ConnectionError:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Try: ollama serve"
            )
        except requests.Timeout:
            raise EmbeddingError(
                f"Ollama request timed out after {self._timeout[1]}s."
            )

        if not response.ok:
            raise EmbeddingError(
                f"Ollama returned {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        vec = np.array(data["embedding"], dtype=np.float64)

        if self._dimension is None:
            self._dimension = len(vec)

        return vec

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe with empty string to detect dimension
            try:
                vec = self.embed("dimension probe")
                self._dimension = len(vec)
            except EmbeddingError:
                return 768  # Assume nomic default
        return self._dimension

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"
