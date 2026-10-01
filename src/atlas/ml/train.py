"""End-to-end training pipeline: SQL extract → features → model selection → eval → registry."""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from atlas.config import get_settings
from atlas.data.generate import CATEGORIES, PRIORITIES
from atlas.db import read_only_query
from atlas.ml import anomaly
from atlas.ml.classifier import (
    category_candidates,
    clean_text,
    predict_with_confidence,
    priority_model,
)
from atlas.ml.registry import ModelRegistry
from atlas.ml.tracking import Tracker

log = logging.getLogger(__name__)

TRAINING_SQL = """
SELECT t.ticket_id, t.created_at, t.channel, t.body, t.category, t.priority,
       c.tier
FROM tickets t
JOIN customers c ON c.customer_id = t.customer_id
ORDER BY t.created_at
"""


def load_training_frame() -> pd.DataFrame:
    df = read_only_query(TRAINING_SQL)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["text"] = clean_text(df["body"])
    return df


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df[["ticket_id", "category"]]).values.tobytes()
                          ).hexdigest()[:12]


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split: train on the past, test on the most recent tickets (no leakage)."""
    cutoff = df["created_at"].quantile(1 - test_frac)
    return df[df["created_at"] < cutoff].copy(), df[df["created_at"] >= cutoff].copy()


def _clf_metrics(y_true, y_pred, conf, labels, threshold: float) -> dict:
    auto = conf >= threshold
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "auto_route_rate": round(float(auto.mean()), 4),
        "auto_route_accuracy": round(float((np.asarray(y_true)[auto] == y_pred[auto]).mean()), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def train_category(train: pd.DataFrame, test: pd.DataFrame, tracker: Tracker,
                   threshold: float) -> tuple[object, dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: dict[str, float] = {}
    for name, pipe in category_candidates().items():
        with tracker.run(f"category/{name}", nested=True):
            s = cross_val_score(pipe, train["text"], train["category"], cv=cv,
                                scoring="f1_macro", n_jobs=-1)
            scores[name] = float(s.mean())
            tracker.log_params({"candidate": name})
            tracker.log_metrics({"cv_macro_f1_mean": s.mean(), "cv_macro_f1_std": s.std()})
            log.info("candidate %-22s cv macro-F1 = %.4f ± %.4f", name, s.mean(), s.std())

    best_name = max(scores, key=scores.get)
    best = category_candidates()[best_name].fit(train["text"], train["category"])
    y_pred, conf = predict_with_confidence(best, test["text"])
    metrics = _clf_metrics(test["category"], y_pred, conf, CATEGORIES, threshold)
    metrics.update({"selected_model": best_name, "cv_scores": scores})
    return best, metrics


def train_priority(train: pd.DataFrame, test: pd.DataFrame, threshold: float) -> tuple[object, dict]:
    cols = ["text", "tier", "channel", "category"]
    model = priority_model().fit(train[cols], train["priority"])
    y_pred, conf = predict_with_confidence(model, test[cols])
    metrics = _clf_metrics(test["priority"], y_pred, conf, PRIORITIES, threshold)
    # Business-critical: how many truly urgent tickets do we catch?
    urgent = test["priority"].to_numpy() == "urgent"
    metrics["urgent_recall"] = round(float((y_pred[urgent] == "urgent").mean()), 4)
    return model, metrics


def train_anomaly(tracker: Tracker) -> tuple[anomaly.AnomalyDetector, dict]:
    """Champion/challenger: evaluate both detectors on labelled incident days, promote the best."""
    settings = get_settings()
    wide = anomaly.daily_matrix(read_only_query(anomaly.DAILY_COUNTS_SQL))
    feats = anomaly.build_features(wide)
    incidents = pd.read_csv(settings.data_dir / "incidents.csv", parse_dates=["date"])
    truth = set(incidents["date"])

    results, detectors = {}, {}
    for method in ("robust_z", "isolation_forest"):
        det = anomaly.AnomalyDetector(method=method).fit(feats)
        results[method] = anomaly.detection_metrics(det.score(feats)["is_anomaly"], truth)
        detectors[method] = det
        tracker.log_metrics({f"anomaly_{method}_{k}": v for k, v in results[method].items()})

    champion = max(results, key=lambda m: results[m]["f1"])
    metrics = {**results, "champion": champion,
               "top_anomalies": anomaly.top_anomalies(detectors[champion].score(feats)),
               "days": len(feats)}
    return detectors[champion], metrics


def run_training() -> dict:
    settings = get_settings()
    tracker = Tracker(settings.mlflow_tracking_uri)
    registry = ModelRegistry(settings.models_dir)
    df = load_training_frame()
    train, test = temporal_split(df)
    fp = fingerprint(df)
    thr = settings.auto_route_threshold

    with tracker.run("training-pipeline", tags={"data_fingerprint": fp}):
        tracker.log_params({"n_train": len(train), "n_test": len(test), "split": "temporal-80/20",
                            "auto_route_threshold": thr})

        cat_model, cat_metrics = train_category(train, test, tracker, thr)
        tracker.log_metrics({f"category_{k}": v for k, v in cat_metrics.items()
                             if isinstance(v, float)})
        cat_mv = registry.register("category", cat_model,
                                   {k: v for k, v in cat_metrics.items() if isinstance(v, float)},
                                   {"selected_model": cat_metrics["selected_model"]}, fp)

        pri_model, pri_metrics = train_priority(train, test, thr)
        tracker.log_metrics({f"priority_{k}": v for k, v in pri_metrics.items()
                             if isinstance(v, float)})
        pri_mv = registry.register("priority", pri_model,
                                   {k: v for k, v in pri_metrics.items() if isinstance(v, float)},
                                   {"features": "tfidf(text)+onehot(tier,channel,category)"}, fp)

        det, an_metrics = train_anomaly(tracker)
        an_mv = registry.register("anomaly", det, an_metrics[an_metrics["champion"]],
                                  {"method": det.method, "threshold": det.threshold}, fp)

        # Reference distribution for drift monitoring (PSI) in production.
        reference = {"category": train["category"].value_counts(normalize=True).to_dict(),
                     "priority": train["priority"].value_counts(normalize=True).to_dict()}

        report = {
            "data_fingerprint": fp,
            "n_train": len(train), "n_test": len(test),
            "category": {**cat_metrics, "version": cat_mv.version},
            "priority": {**pri_metrics, "version": pri_mv.version},
            "anomaly": {**an_metrics, "version": an_mv.version},
            "reference_distribution": reference,
        }
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        path = settings.reports_dir / "training_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        (settings.models_dir / "reference_distribution.json").write_text(
            json.dumps(reference, indent=2), encoding="utf-8")
        tracker.log_artifact(path)
    return report
