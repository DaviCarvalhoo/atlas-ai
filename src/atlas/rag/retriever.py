"""Vector store (ChromaDB) indexing and hybrid retrieval (dense + keyword, fused with RRF)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import chromadb
import joblib
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sklearn.feature_extraction.text import TfidfVectorizer

from atlas.config import Settings, get_settings
from atlas.rag.chunking import Chunk, load_corpus
from atlas.rag.embeddings import build_embedder

log = logging.getLogger(__name__)

COLLECTION = "knowledge_base"
RRF_K = 60

Mode = Literal["dense", "keyword", "hybrid"]


@dataclass
class RetrievedChunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    score: float

    def citation(self) -> str:
        return f"{self.source} § {self.section}"


def _client(settings: Settings) -> chromadb.ClientAPI:
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.vector_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def build_index(settings: Settings | None = None, extra_corpus: list[str] | None = None) -> dict:
    """(Re)build the vector index. ``extra_corpus`` (e.g. ticket texts) enriches LSA semantics."""
    settings = settings or get_settings()
    chunks = load_corpus(settings.knowledge_base_dir)
    texts = [c.embedding_text for c in chunks]
    embedder = build_embedder(settings, corpus=texts + (extra_corpus or []))
    vectors = embedder.embed(texts)

    client = _client(settings)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embedder": embedder.name}
    )
    col.add(
        ids=[c.id for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "title": c.title, "section": c.section} for c in chunks],
    )

    # Char n-grams act as a light Portuguese stemmer ("prazo"/"prazos", "devolver"/"devolução").
    keyword = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, strip_accents="unicode"
    )
    keyword_matrix = keyword.fit_transform(texts)
    (settings.artifacts_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": keyword, "matrix": keyword_matrix, "chunks": chunks},
        settings.artifacts_dir / "embeddings" / "keyword.joblib",
    )
    log.info(
        "Indexed %d chunks from %s with %s embeddings",
        len(chunks),
        settings.knowledge_base_dir,
        embedder.name,
    )
    return {
        "chunks": len(chunks),
        "documents": len({c.source for c in chunks}),
        "embedder": embedder.name,
        "dims": int(vectors.shape[1]),
    }


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = build_embedder(self.settings)
        self.collection = _client(self.settings).get_collection(COLLECTION)
        kw = joblib.load(self.settings.artifacts_dir / "embeddings" / "keyword.joblib")
        self.keyword_vectorizer, self.keyword_matrix = kw["vectorizer"], kw["matrix"]
        self.chunks: list[Chunk] = kw["chunks"]
        self.by_id = {c.id: c for c in self.chunks}

    def _dense(self, query: str, n: int) -> list[tuple[str, float]]:
        vec = self.embedder.embed([query])[0].tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=min(n, len(self.chunks)))
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def _keyword(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = (
            (self.keyword_matrix @ self.keyword_vectorizer.transform([query]).T).toarray().ravel()
        )
        top = np.argsort(-scores)[:n]
        return [(self.chunks[i].id, float(scores[i])) for i in top if scores[i] > 0]

    def search(
        self, query: str, k: int = 4, mode: Mode = "hybrid", boost_sources: set[str] | None = None
    ) -> list[RetrievedChunk]:
        """``boost_sources``: documents favoured by metadata (e.g. the ticket's predicted
        category). Applied as an extra ranking in the fusion, so it nudges but never filters."""
        if mode == "dense":
            ranked = self._dense(query, k)
        elif mode == "keyword":
            ranked = self._keyword(query, k)
        else:  # Reciprocal Rank Fusion of both rankings
            fused: dict[str, float] = {}
            for ranking in (self._dense(query, 10), self._keyword(query, 10)):
                for rank, (cid, _) in enumerate(ranking):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if boost_sources:
                for cid in fused:
                    if self.by_id[cid].source in boost_sources:
                        fused[cid] += 1.0 / (RRF_K + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        out = []
        for cid, score in ranked[:k]:
            c = self.by_id[cid]
            out.append(RetrievedChunk(c.id, c.source, c.title, c.section, c.text, round(score, 4)))
        return out


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()
