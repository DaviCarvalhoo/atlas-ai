"""Retrieval evaluation on a golden set: hit@k, MRR, per retrieval mode."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.rag.retriever import Retriever

GOLDEN = Path(__file__).parent / "golden" / "rag.jsonl"


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_retrieval(retriever: Retriever, k: int = 5, triage_service=None) -> dict:
    """Compare retrieval modes. With ``triage_service``, also evaluates hybrid retrieval boosted
    by the predicted ticket category (metadata-aware retrieval, as used by the agent)."""
    from atlas.agent.graph import CATEGORY_BOOST_MIN_CONFIDENCE, CATEGORY_SOURCES

    golden = load_golden()
    boosts = {}
    if triage_service is not None:
        for item, t in zip(golden, triage_service.classify_many([g["question"] for g in golden]),
                           strict=True):
            boosts[item["question"]] = (CATEGORY_SOURCES.get(t.category)
                                        if t.category_confidence >= CATEGORY_BOOST_MIN_CONFIDENCE
                                        else None)
    modes = ["keyword", "dense", "hybrid"] + (["hybrid+triage"] if boosts else [])
    report = {}
    for mode in modes:
        hits = {1: 0, 3: 0, k: 0}
        rr, misses = 0.0, []
        for item in golden:
            results = retriever.search(item["question"], k=k, mode=mode.split("+")[0],
                                       boost_sources=boosts.get(item["question"])
                                       if mode == "hybrid+triage" else None)
            sources = [r.source for r in results]
            rank = sources.index(item["source"]) + 1 if item["source"] in sources else None
            for cut in hits:
                hits[cut] += int(rank is not None and rank <= cut)
            rr += 1.0 / rank if rank else 0.0
            if rank != 1:
                misses.append({"question": item["question"], "expected": item["source"],
                               "got": sources[:3]})
        n = len(golden)
        report[mode] = {**{f"hit@{c}": round(v / n, 3) for c, v in hits.items()},
                        "mrr": round(rr / n, 3), "n": n, "top1_misses": misses[:5]}
    return report
