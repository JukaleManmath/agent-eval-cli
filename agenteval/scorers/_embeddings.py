from __future__ import annotations

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


_model = None


def _get_model():
    global _model
    if not ML_AVAILABLE:
        raise ImportError(
            "ML dependencies not installed. Run: pip install agent-eval-cli[ml]"
        )
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    model = _get_model()
    return list(model.encode(texts, normalize_embeddings=True))
