"""Ticket triage models: category (text only) and priority (text + customer tier + channel)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from atlas.security.pii import mask_pii

ORDER_NUMBER = re.compile(r"\b\d{5}\b")


def clean_text(texts: pd.Series | list[str]) -> list[str]:
    """Normalise text exactly as at inference time: mask PII, lowercase, mask order numbers."""
    return [ORDER_NUMBER.sub(" num_pedido ", mask_pii(str(t)).text.lower()) for t in texts]


def _text_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=40000,
                ),
            ),
        ]
    )


def category_candidates(seed: int = 42) -> dict[str, Pipeline]:
    """Candidate models compared in the training pipeline (best one is promoted)."""
    return {
        "dummy_most_frequent": Pipeline(
            [
                ("features", TfidfVectorizer()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "tfidf_complement_nb": Pipeline(
            [
                ("features", _text_features()),
                ("clf", ComplementNB(alpha=0.3)),
            ]
        ),
        "tfidf_logreg": Pipeline(
            [
                ("features", _text_features()),
                (
                    "clf",
                    LogisticRegression(
                        C=4.0, max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
    }


def priority_model(seed: int = 42) -> Pipeline:
    features = ColumnTransformer(
        [
            ("text", _text_features(), "text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["tier", "channel", "category"]),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=2.0, max_iter=3000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )


def predict_with_confidence(model: Pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)
    idx = proba.argmax(axis=1)
    return model.classes_[idx], proba[np.arange(len(idx)), idx]
