"""Data preparation: generate → validate → persist (SQL + parquet-free CSV snapshot)."""

from __future__ import annotations

import logging

import pandas as pd

from atlas.config import get_settings
from atlas.data.generate import CATEGORIES, PRIORITIES, generate
from atlas.db import create_schema, get_engine, load_frames

log = logging.getLogger(__name__)


def validate(tickets: pd.DataFrame) -> dict:
    """Lightweight data-quality checks (would be Great Expectations / pandera in a bigger setup)."""
    issues = {
        "null_body": int(tickets["body"].isna().sum()),
        "empty_body": int((tickets["body"].str.strip() == "").sum()),
        "unknown_category": int((~tickets["category"].isin(CATEGORIES)).sum()),
        "unknown_priority": int((~tickets["priority"].isin(PRIORITIES)).sum()),
        "duplicate_ids": int(tickets["ticket_id"].duplicated().sum()),
    }
    if any(issues.values()):
        raise ValueError(f"Data quality check failed: {issues}")
    return {
        "rows": len(tickets),
        "category_distribution": tickets["category"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict(),
        "priority_distribution": tickets["priority"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict(),
        "avg_body_chars": round(float(tickets["body"].str.len().mean()), 1),
    }


def build_database(seed: int = 42) -> dict:
    settings = get_settings()
    ds = generate(seed=seed)
    profile = validate(ds.tickets)

    engine = get_engine()
    create_schema(engine)
    load_frames(engine, {"customers": ds.customers, "orders": ds.orders, "tickets": ds.tickets})

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    ds.incidents.to_csv(settings.data_dir / "incidents.csv", index=False)
    log.info(
        "Database built: %s tickets, %s orders, %s customers",
        len(ds.tickets),
        len(ds.orders),
        len(ds.customers),
    )
    return {
        **profile,
        "orders": len(ds.orders),
        "customers": len(ds.customers),
        "incident_days": len(ds.incidents),
    }
