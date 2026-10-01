"""Database access: engine factory, schema bootstrap and *read-only* query execution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from atlas.config import get_settings

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def _statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def create_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_customer_safe"))
        for table in ("tickets", "orders", "customers"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        for stmt in _statements(SCHEMA_PATH.read_text(encoding="utf-8")):
            conn.execute(text(stmt))


def load_frames(engine: Engine, frames: dict[str, pd.DataFrame]) -> None:
    """Append dataframes into the (already created) schema, parents first."""
    for name in ("customers", "orders", "tickets"):
        frames[name].to_sql(name, engine, if_exists="append", index=False, chunksize=2000)


def read_only_query(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    """Run a query inside a read-only transaction (defense in depth on top of the SQL guard)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            conn.exec_driver_sql("PRAGMA query_only = ON")
        elif engine.dialect.name == "postgresql":
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        try:
            return pd.read_sql_query(text(sql), conn, params=params or {})
        finally:
            if engine.dialect.name == "sqlite":
                conn.exec_driver_sql("PRAGMA query_only = OFF")
            conn.rollback()
