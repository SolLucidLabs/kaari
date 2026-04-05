"""Base embedding provider interface."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract base for embedding providers.

    All providers must implement embed() → numpy array.
    Kaari core doesn't care which provider is used.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text into a dense vector.

        Args:
            text: Input text to embed.

        Returns:
            numpy array (typically 768-dim for nomic, 1536-dim for OpenAI).

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension (e.g., 768 for nomic-embed-text)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/config."""
        ...


class EmbeddingError(Exception):
    """Raised when embedding fails."""
    pass
