"""Inference service for the triage models (loaded once from the registry's production alias)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from atlas.config import get_settings
from atlas.ml.classifier import clean_text, predict_with_confidence
from atlas.ml.registry import ModelRegistry


@dataclass
class Triage:
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    auto_routed: bool
    model_versions: dict[str, int]


class TriageService:
    def __init__(self):
        settings = get_settings()
        self.registry = ModelRegistry(settings.models_dir)
        self.threshold = settings.auto_route_threshold
        self.category_model = self.registry.load("category")
        self.priority_model = self.registry.load("priority")
        self.versions = {
            "category": self.registry.production_version("category"),
            "priority": self.registry.production_version("priority"),
        }

    def classify_many(
        self, texts: list[str], tiers: list[str] | None = None, channels: list[str] | None = None
    ) -> list[Triage]:
        n = len(texts)
        cleaned = clean_text(texts)
        cats, cat_conf = predict_with_confidence(self.category_model, cleaned)
        frame = pd.DataFrame(
            {
                "text": cleaned,
                "tier": tiers or ["standard"] * n,
                "channel": channels or ["chat"] * n,
                "category": cats,
            }
        )
        pris, pri_conf = predict_with_confidence(self.priority_model, frame)
        return [
            Triage(
                str(c),
                round(float(cc), 3),
                str(p),
                round(float(pc), 3),
                bool(cc >= self.threshold),
                self.versions,
            )
            for c, cc, p, pc in zip(cats, cat_conf, pris, pri_conf, strict=True)
        ]

    def classify(self, text: str, tier: str = "standard", channel: str = "chat") -> Triage:
        return self.classify_many([text], [tier], [channel])[0]


@lru_cache
def get_triage_service() -> TriageService:
    return TriageService()
