"""Atlas support agent as a LangGraph state machine.

    guard ──blocked──► END
      │
    triage (sklearn) ─► route (LLM | heuristic)
                          ├─ order_lookup ─► retrieve ─┐
                          ├─ analytics ────────────────┼─► answer ─► END
                          └─ retrieve ─────────────────┘

All dependencies are injected, so the graph is unit-testable with fakes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from atlas.agent import tools
from atlas.llm.prompts import (
    ANSWER_SYSTEM,
    PROMPT_VERSION,
    ROUTER_SYSTEM,
    SQL_SYSTEM,
    answer_user_prompt,
)
from atlas.rag.retriever import RetrievedChunk
from atlas.security.guardrails import check_input
from atlas.security.pii import mask_pii
from atlas.security.sql_guard import UnsafeSQLError

log = logging.getLogger(__name__)

Intent = Literal["order_lookup", "analytics", "knowledge"]

ANALYTICS_HINTS = (
    "quantos",
    "quantas",
    "taxa",
    "media",
    "percentual",
    "ranking",
    "volume",
    "por categoria",
    "por canal",
    "por prioridade",
    "por mes",
    "csat",
    "sla",
    "tendencia",
    "evolucao",
    "transportadora com",
    "mais atras",
    "relatorio",
)


# Triage category → knowledge-base documents most likely to hold the answer.
CATEGORY_SOURCES = {
    "billing": {"politica-de-reembolso.md", "pagamentos-e-parcelamento.md"},
    "shipping": {"prazos-e-entregas.md"},
    "technical": {"suporte-tecnico-app.md"},
    "account": {"conta-e-seguranca.md", "lgpd-e-privacidade.md"},
    "returns": {"politica-de-trocas-e-devolucoes.md"},
}
CATEGORY_BOOST_MIN_CONFIDENCE = 0.75


class AgentState(TypedDict, total=False):
    question: str
    customer_id: int | None
    pii_found: dict[str, int]
    blocked: bool
    intent: Intent
    order_id: int | None
    order: dict | None
    triage: dict
    chunks: list[RetrievedChunk]
    sql: str | None
    rows: list[dict]
    answer: str
    citations: list[str]
    steps: list[str]
    llm: dict


class SupportAgent:
    def __init__(self, llm, retriever, triage_service, k: int = 4):
        self.llm, self.retriever, self.triage_service, self.k = llm, retriever, triage_service, k
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("guard", self.guard)
        g.add_node("triage", self.triage)
        g.add_node("route", self.route)
        g.add_node("order_lookup", self.order_lookup)
        g.add_node("analytics", self.analytics)
        g.add_node("retrieve", self.retrieve)
        g.add_node("answer", self.answer)

        g.set_entry_point("guard")
        g.add_conditional_edges(
            "guard",
            lambda s: "end" if s.get("blocked") else "triage",
            {"end": END, "triage": "triage"},
        )
        g.add_edge("triage", "route")
        g.add_conditional_edges(
            "route",
            lambda s: s["intent"],
            {"order_lookup": "order_lookup", "analytics": "analytics", "knowledge": "retrieve"},
        )
        g.add_edge("order_lookup", "retrieve")
        g.add_edge("analytics", "answer")
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def invoke(self, question: str, customer_id: int | None = None) -> AgentState:
        return self.graph.invoke({"question": question, "customer_id": customer_id, "steps": []})

    # ------------------------------------------------------------------ nodes
    def guard(self, state: AgentState) -> AgentState:
        masked = mask_pii(state["question"])
        verdict = check_input(masked.text)
        steps = [*state["steps"], f"guard: pii={masked.found or 'none'} allowed={verdict.allowed}"]
        if not verdict.allowed:
            return {
                "question": masked.text,
                "pii_found": masked.found,
                "blocked": True,
                "steps": steps,
                "citations": [],
                "intent": "knowledge",
                "answer": "Não posso processar esta solicitação por motivos de segurança "
                f"({verdict.reason}). Um atendente humano pode ajudar.",
            }
        return {
            "question": masked.text,
            "pii_found": masked.found,
            "blocked": False,
            "steps": steps,
        }

    def triage(self, state: AgentState) -> AgentState:
        t = self.triage_service.classify(state["question"])
        return {
            "triage": asdict(t),
            "steps": [
                *state["steps"],
                f"triage: {t.category}/{t.priority} ({t.category_confidence})",
            ],
        }

    def route(self, state: AgentState) -> AgentState:
        q = state["question"]
        order_id = tools.extract_order_id(q)
        intent: Intent | None = None
        if not self.llm.offline:
            try:
                resp = self.llm.complete(
                    ROUTER_SYSTEM, q, max_tokens=200, json_mode=True, task="route"
                )
                data = _parse_json(resp.text)
                if data.get("intent") in ("order_lookup", "analytics", "knowledge"):
                    intent = data["intent"]
                    order_id = data.get("order_id") or order_id
            except Exception as exc:  # degrade gracefully to the heuristic router
                log.warning("LLM routing failed, falling back to heuristics: %s", exc)
        if intent is None:
            intent = self._heuristic_route(q, order_id)
        return {
            "intent": intent,
            "order_id": order_id,
            "steps": [*state["steps"], f"route: {intent} (order_id={order_id})"],
        }

    @staticmethod
    def _heuristic_route(question: str, order_id: int | None) -> Intent:
        q = tools.normalize(question)
        if any(h in q for h in ANALYTICS_HINTS) and tools.analytics_query_for(question):
            return "analytics"
        if order_id is not None:
            return "order_lookup"
        return "knowledge"

    def order_lookup(self, state: AgentState) -> AgentState:
        order = (
            tools.lookup_order(state["order_id"], state.get("customer_id"))
            if state.get("order_id")
            else None
        )
        return {
            "order": order,
            "steps": [*state["steps"], f"order_lookup: {'found' if order else 'not found'}"],
        }

    def analytics(self, state: AgentState) -> AgentState:
        q, sql, df = state["question"], None, pd.DataFrame()
        try:
            if not self.llm.offline:
                resp = self.llm.complete(
                    SQL_SYSTEM, q, max_tokens=500, json_mode=True, task="text_to_sql"
                )
                sql, df = tools.run_sql(_parse_json(resp.text)["sql"])
            else:
                name = tools.analytics_query_for(q)
                if name:
                    sql, df = tools.run_named_analytics(name)
        except (UnsafeSQLError, KeyError, ValueError) as exc:
            log.warning("text-to-SQL rejected: %s", exc)
            return {
                "sql": None,
                "rows": [],
                "steps": [*state["steps"], f"analytics: rejected ({exc})"],
            }
        return {
            "sql": sql,
            "rows": df.head(50).to_dict(orient="records"),
            "steps": [*state["steps"], f"analytics: {len(df)} rows"],
        }

    def retrieve(self, state: AgentState) -> AgentState:
        query = state["question"]
        if state.get("order"):  # enrich the query so the policy for this situation is retrieved
            o = state["order"]
            query += (
                " atraso entrega" if (o.get("days_late") or 0) > 0 else f" pedido {o['status']}"
            )
        triage = state.get("triage") or {}
        boost = (
            CATEGORY_SOURCES.get(triage.get("category"))
            if triage.get("category_confidence", 0) >= CATEGORY_BOOST_MIN_CONFIDENCE
            else None
        )
        chunks = self.retriever.search(query, k=self.k, boost_sources=boost)
        return {
            "chunks": chunks,
            "steps": [*state["steps"], f"retrieve: {[c.citation() for c in chunks[:3]]}"],
        }

    def answer(self, state: AgentState) -> AgentState:
        context, citations = self._context(state)
        llm_meta: dict[str, Any] = {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "prompt_version": PROMPT_VERSION,
        }
        if state["intent"] == "analytics":
            text = _analytics_answer(state.get("rows", []), state.get("sql"))
        elif self.llm.offline:
            text = _extractive_answer(
                state["question"], state.get("order"), state.get("chunks", [])
            )
        else:
            resp = self.llm.complete(
                ANSWER_SYSTEM,
                answer_user_prompt(state["question"], context),
                max_tokens=700,
                task="answer",
            )
            text = resp.text
            llm_meta.update(
                latency_ms=resp.latency_ms,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )
        return {
            "answer": text,
            "citations": citations,
            "llm": llm_meta,
            "steps": [*state["steps"], "answer: done"],
        }

    @staticmethod
    def _context(state: AgentState) -> tuple[list[str], list[str]]:
        blocks, cites = [], []
        if order := state.get("order"):
            blocks.append(
                "Dados do pedido (banco de dados): " + json.dumps(order, ensure_ascii=False)
            )
            cites.append(f"orders#{order['order_id']}")
        for c in state.get("chunks", []):
            blocks.append(f"{c.title} — {c.section}: {c.text}")
            cites.append(c.citation())
        return blocks, cites


# ---------------------------------------------------------------------- helpers
def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def _sentences(text: str) -> list[str]:
    # Split on sentence ends, but not after common abbreviations such as "art." or "ex.".
    parts = re.split(r"(?<!\bart\.)(?<!\bex\.)(?<=[.!?])\s+|\n+", text.replace("**", ""))
    cleaned = [re.sub(r"^\s*(\d+\.|[-*|#])\s*", "", p).strip() for p in parts]
    return [p for p in cleaned if len(p) > 15 and not p.endswith(":")]


def _extractive_answer(question: str, order: dict | None, chunks: list[RetrievedChunk]) -> str:
    """Offline, deterministic answer: order facts + best-matching policy sentences + citations."""
    lines = []
    if order:
        status = {
            "delivered": "entregue",
            "shipped": "em transporte",
            "processing": "processando",
            "cancelled": "cancelado",
            "returned": "devolvido",
        }.get(order["status"], order["status"])
        line = (
            f"O pedido {order['order_id']} está **{status}** (transportadora {order['carrier']}, "
            f"previsão {str(order['estimated_delivery'])[:10]})"
        )
        if order.get("days_late"):
            line += f", entregue com {order['days_late']} dia(s) de atraso"
        lines.append(line + ".")
    q_tokens = tools.tokens(question)
    scored = []
    for rank, c in enumerate(chunks[:3]):
        for s in _sentences(c.text):
            overlap = len(q_tokens & tools.tokens(s + " " + c.section))
            scored.append((overlap / (1 + 0.5 * rank), rank, s))
    best = [x for x in sorted(scored, key=lambda x: -x[0]) if x[0] > 0][:2]
    for _, rank, sentence in best:
        lines.append(f"{sentence} [{rank + 1 + (1 if order else 0)}]")
    if not lines:
        return (
            "Não encontrei essa informação na base de conhecimento. "
            "Vou encaminhar para um atendente humano."
        )
    return " ".join(lines)


def _analytics_answer(rows: list[dict], sql: str | None) -> str:
    if not rows:
        return (
            "Não consegui responder com segurança a essa pergunta analítica. "
            "Tente reformular (ex.: 'taxa de violação de SLA por prioridade')."
        )
    cols = list(rows[0])
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += [
        "| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |"
        for r in rows[:20]
    ]
    return (
        f"Resultado ({len(rows)} linhas):\n\n"
        + "\n".join(table)
        + f"\n\nSQL executado:\n```sql\n{sql}\n```"
    )
