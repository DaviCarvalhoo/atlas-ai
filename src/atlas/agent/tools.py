"""Tools available to the agent. Each one is small, typed and safe by construction."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from atlas.analytics import load_queries
from atlas.db import read_only_query
from atlas.security.sql_guard import validate_sql

ORDER_SQL = """
SELECT o.order_id, o.status, o.created_at, o.estimated_delivery, o.delivered_at, o.carrier,
       o.payment_method, o.total_value, c.tier
FROM orders o
JOIN v_customer_safe c ON c.customer_id = o.customer_id
WHERE o.order_id = :order_id
"""

ORDER_ID = re.compile(r"\b(1\d{4})\b")

STOPWORDS = set("""a o as os de da do das dos e é em um uma para por com no na nos nas que qual quais
meu minha meus minhas se eu voce vocês voces como quando quanto quantos quantas ao à mais ou já
ja tem ter foi ser está esta isso esse essa sobre posso pode consigo""".split())

# Offline intent → named analytics query (used when no LLM is configured for text-to-SQL).
ANALYTICS_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("sla", "prazo de resolucao", "estouro", "violacao"), "sla_breaches_by_priority"),
    (("transportadora", "atraso", "atrasos", "entrega"), "carrier_delay_ranking"),
    (("mes", "mensal", "evolucao", "crescimento", "tendencia"), "monthly_volume"),
    (("csat", "satisfacao", "canal", "canais"), "csat_by_channel"),
    (("recorrente", "reincidente", "mais chamados", "clientes que mais"), "top_contacting_customers"),
    (("categoria", "categorias", "volume", "tickets", "chamados"), "tickets_by_category"),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", normalize(text)) if t not in STOPWORDS and len(t) > 1}


def extract_order_id(text: str) -> int | None:
    m = ORDER_ID.search(text)
    return int(m.group(1)) if m else None


def lookup_order(order_id: int, customer_id: int | None = None) -> dict | None:
    """Parametrised lookup (no string interpolation → no SQL injection). Ownership is enforced
    when the caller is an authenticated customer."""
    sql = ORDER_SQL + (" AND o.customer_id = :customer_id" if customer_id is not None else "")
    params = {"order_id": order_id, **({"customer_id": customer_id} if customer_id else {})}
    df = read_only_query(sql, params)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    eta, delivered = pd.to_datetime(row["estimated_delivery"]), row["delivered_at"]
    row["days_late"] = (pd.to_datetime(delivered) - eta).days if pd.notna(delivered) else None
    return {k: (None if pd.isna(v) else (str(v) if not isinstance(v, int | float) else v))
            for k, v in row.items()}


def analytics_query_for(question: str) -> str | None:
    q = normalize(question)
    for keywords, name in ANALYTICS_KEYWORDS:
        if any(k in q for k in keywords):
            return name
    return None


def run_sql(sql: str) -> tuple[str, pd.DataFrame]:
    safe = validate_sql(sql)
    return safe, read_only_query(safe)


def run_named_analytics(name: str) -> tuple[str, pd.DataFrame]:
    return run_sql(load_queries()[name].sql)
