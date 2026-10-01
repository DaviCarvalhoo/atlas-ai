"""Pluggable embedding providers.

- ``lsa``: TF-IDF + TruncatedSVD (Latent Semantic Analysis) fitted on the domain corpus. Fully
  offline and deterministic — the default for CI and demos.
- ``sentence-transformers``: multilingual MiniLM, local neural embeddings (optional extra).
- ``openai``: ``text-embedding-3-small`` via API (also works for Azure deployments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from atlas.config import Settings


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 192, seed: int = 42):
        self.dims = dims
        self.seed = seed
        self.pipeline = None

    def fit(self, corpus: list[str]) -> LSAEmbedder:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", min_df=1, lowercase=True
        )
        n_features = tfidf.fit(corpus).transform(corpus).shape[1]
        dims = max(2, min(self.dims, len(corpus) - 1, n_features - 1))
        self.pipeline = make_pipeline(
            tfidf, TruncatedSVD(dims, random_state=self.seed), Normalizer(copy=False)
        )
        self.pipeline.fit(corpus)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LSAEmbedder must be fitted (or loaded) before use")
        return self.pipeline.transform(texts).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> LSAEmbedder:
        emb = cls()
        emb.pipeline = joblib.load(path)
        return emb


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def build_embedder(settings: Settings, corpus: list[str] | None = None) -> Embedder:
    """Return the configured embedder. LSA is fitted on ``corpus`` or loaded from disk."""
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.st_model)
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    path = settings.artifacts_dir / "embeddings" / "lsa.joblib"
    if corpus is not None:
        emb = LSAEmbedder().fit(corpus)
        emb.save(path)
        return emb
    return LSAEmbedder.load(path)
