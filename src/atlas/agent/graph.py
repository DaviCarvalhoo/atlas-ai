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
    INSIGHT_SYSTEM,
    PROMPT_VERSION,
    ROUTER_SYSTEM,
    SQL_REPAIR,
    answer_user_prompt,
)
from atlas.llm.providers import LLMError
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

# Generous caps: reasoning models (e.g. gpt-oss, o-series) spend tokens thinking before answering.
MAX_TOKENS = {"route": 1500, "sql": 3000, "answer": 3000}


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
                    ROUTER_SYSTEM, q, max_tokens=MAX_TOKENS["route"], json_mode=True, task="route"
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
        q = state["question"]
        if self.llm.offline:
            return self._template_analytics(state, reason="offline")
        try:
            sql, df, attempts = self._text_to_sql(q)
        except UnsafeSQLError as exc:  # a security violation is never retried or worked around
            log.warning("text-to-SQL rejected: %s", exc)
            return {
                "sql": None,
                "rows": [],
                "steps": [*state["steps"], f"analytics: rejected ({exc})"],
            }
        except LLMError as exc:  # provider unavailable → curated queries still work
            log.warning("LLM unavailable for text-to-SQL: %s", exc)
            return self._template_analytics(state, reason="llm unavailable")
        if sql is None:
            return {
                "sql": None,
                "rows": [],
                "steps": [*state["steps"], f"analytics: failed after {attempts} attempts"],
            }
        return {
            "sql": sql,
            "rows": df.head(50).to_dict(orient="records"),
            "steps": [*state["steps"], f"analytics: {len(df)} rows (llm sql, attempts={attempts})"],
        }

    def _text_to_sql(self, question: str, max_attempts: int = 2):
        """Generate → guard → execute; on a database error, feed the error back to the model
        once so it can repair the query (self-correction loop)."""
        prompt = question
        for attempt in range(1, max_attempts + 1):
            resp = self.llm.complete(
                tools.sql_system_prompt(),
                prompt,
                max_tokens=MAX_TOKENS["sql"],
                json_mode=True,
                task="text_to_sql" if attempt == 1 else "sql_repair",
            )
            candidate = _parse_json(resp.text).get("sql", "")
            try:
                sql, df = tools.run_sql(candidate)
                return sql, df, attempt
            except UnsafeSQLError:
                raise
            except Exception as exc:
                error = str(exc).splitlines()[0][:300]
                log.warning("SQL attempt %d failed: %s", attempt, error)
                prompt = SQL_REPAIR.format(question=question, sql=candidate, error=error)
        return None, None, max_attempts

    def _template_analytics(self, state: AgentState, reason: str) -> AgentState:
        sql, df = None, pd.DataFrame()
        if name := tools.analytics_query_for(state["question"]):
            sql, df = tools.run_named_analytics(name)
        return {
            "sql": sql,
            "rows": df.head(50).to_dict(orient="records"),
            "steps": [*state["steps"], f"analytics: {len(df)} rows (template, {reason})"],
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
            if state.get("rows") and not self.llm.offline:
                try:
                    insight = self.llm.complete(
                        INSIGHT_SYSTEM,
                        f"Pergunta: {state['question']}\nResultado (JSON): "
                        + json.dumps(state["rows"][:50], ensure_ascii=False, default=str),
                        max_tokens=MAX_TOKENS["answer"],
                        task="insight",
                    )
                    text = f"{insight.text.strip()}\n\n{text}"
                except Exception as exc:
                    log.warning("insight generation failed: %s", exc)
        elif self.llm.offline:
            text = _extractive_answer(
                state["question"], state.get("order"), state.get("chunks", [])
            )
        else:
            try:
                resp = self.llm.complete(
                    ANSWER_SYSTEM,
                    answer_user_prompt(state["question"], context),
                    max_tokens=MAX_TOKENS["answer"],
                    task="answer",
                )
                text = resp.text
                llm_meta.update(
                    latency_ms=resp.latency_ms,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )
            except Exception as exc:  # provider down → still answer, from the same context
                log.warning("LLM answer failed, using extractive fallback: %s", exc)
                text = _extractive_answer(
                    state["question"], state.get("order"), state.get("chunks", [])
                )
                llm_meta["fallback"] = "extractive"
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
