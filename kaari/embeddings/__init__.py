"""Kaari Embeddings — Swappable embedding providers."""
from kaari.embeddings.base import EmbeddingProvider
from kaari.embeddings.ollama import OllamaEmbedding

__all__ = ["EmbeddingProvider", "OllamaEmbedding"]

# Optional providers (import only if dependencies available)
try:
    from kaari.embeddings.openai import OpenAIEmbedding
    __all__.append("OpenAIEmbedding")
except ImportError:
    pass
