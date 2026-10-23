"""Lightweight text embeddings for episodic memory search.

Uses a dependency-free hashing vectorizer (unigrams + bigrams) by default so
the project runs anywhere. If ``sentence-transformers`` is installed and
``DRIVEMEMORY_EMBEDDINGS=sentence-transformers`` is set, a real semantic model
is used instead.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

DIM = 512
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z_]+")
_STOP = {"the", "a", "an", "to", "on", "in", "at", "of", "and", "i", "did", "my", "when", "last", "was", "is"}
_model = None


def _tokens(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
    # light stemming so "took a break" matches "breaks"
    words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def _hash_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for tok in _tokens(text):
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
        v[h % DIM] += 1.0 if (h >> 31) & 1 == 0 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(text: str) -> np.ndarray:
    global _model
    if os.getenv("DRIVEMEMORY_EMBEDDINGS") == "sentence-transformers":
        if _model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency

            _model = SentenceTransformer("all-MiniLM-L6-v2")
        v = _model.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)
    return _hash_embed(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))
