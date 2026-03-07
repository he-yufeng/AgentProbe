"""Semantic similarity — lazy-loads sentence-transformers only when needed."""

from __future__ import annotations

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Semantic comparison requires sentence-transformers. "
                "Install it with: pip install agentprobe[semantic]"
            ) from None
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts. Returns a float in [0, 1]."""
    model = _get_model()
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    score = float(embeddings[0] @ embeddings[1])
    return max(0.0, min(1.0, score))


def texts_match(text_a: str, text_b: str, threshold: float = 0.85, mode: str = "semantic") -> bool:
    """Check if two texts match according to the given mode and threshold."""
    if mode == "exact":
        return text_a.strip() == text_b.strip()
    if mode == "semantic":
        return semantic_similarity(text_a, text_b) >= threshold
    raise ValueError(f"Unknown comparison mode: {mode!r}. Use 'exact' or 'semantic'.")
