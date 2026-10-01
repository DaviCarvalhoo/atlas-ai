"""Incident detection on the daily ticket-volume time series.

Pipeline: daily counts per category (SQL) → seasonal baseline (trailing median, weekday-adjusted)
→ robust z-scores → IsolationForest. A pure statistical rule (max robust-z > threshold) is kept
as a baseline so the ML model has to justify itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest

from atlas.data.generate import CATEGORIES

DAILY_COUNTS_SQL = """
SELECT DATE(created_at) AS day, category, COUNT(*) AS n
FROM tickets
GROUP BY DATE(created_at), category
ORDER BY day
"""


def daily_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = (long_df.assign(day=pd.to_datetime(long_df["day"]))
            .pivot_table(index="day", columns="category", values="n", aggfunc="sum", fill_value=0)
            .reindex(columns=CATEGORIES, fill_value=0)
            .asfreq("D", fill_value=0))
    wide["total"] = wide[CATEGORIES].sum(axis=1)
    return wide


def build_features(wide: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Robust z-score of each series against its trailing, weekday-adjusted baseline."""
    feats = pd.DataFrame(index=wide.index)
    weekday_factor = wide["total"].groupby(wide.index.dayofweek).transform("mean") / wide["total"].mean()
    for col in [*CATEGORIES, "total"]:
        adj = wide[col] / weekday_factor
        med = adj.shift(1).rolling(window, min_periods=7).median()
        mad = (adj.shift(1) - med).abs().rolling(window, min_periods=7).median()
        feats[f"z_{col}"] = ((adj - med) / (1.4826 * mad + 1.0)).fillna(0.0)
    feats["share_max"] = (wide[CATEGORIES].max(axis=1) / wide["total"].clip(lower=1))
    return feats


@dataclass
class AnomalyDetector:
    contamination: float = 0.03
    seed: int = 42
    model: IsolationForest | None = None

    def fit(self, feats: pd.DataFrame) -> AnomalyDetector:
        self.model = IsolationForest(n_estimators=300, contamination=self.contamination,
                                     random_state=self.seed).fit(feats.values)
        return self

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None, "call fit() first"
        z_cols = [f"z_{c}" for c in CATEGORIES]
        out = pd.DataFrame(index=feats.index)
        out["anomaly_score"] = -self.model.score_samples(feats.values)
        out["is_anomaly"] = self.model.predict(feats.values) == -1
        out["driver_category"] = feats[z_cols].idxmax(axis=1).str.removeprefix("z_")
        out["driver_z"] = feats[z_cols].max(axis=1).round(2)
        return out


def zscore_baseline(feats: pd.DataFrame, threshold: float = 4.0) -> pd.Series:
    return feats[[f"z_{c}" for c in CATEGORIES]].max(axis=1) > threshold


def detection_metrics(pred: pd.Series, truth_days: set[pd.Timestamp]) -> dict[str, float]:
    truth = pd.Series(pred.index.isin(list(truth_days)), index=pred.index)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def top_anomalies(scored: pd.DataFrame, k: int = 10) -> list[dict]:
    rows = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(k)
    return [{"day": d.date().isoformat(), "score": round(float(r.anomaly_score), 3),
             "driver_category": r.driver_category, "driver_z": float(r.driver_z)}
            for d, r in rows.iterrows()]

