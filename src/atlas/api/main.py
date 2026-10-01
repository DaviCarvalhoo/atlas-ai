"""Atlas REST API (FastAPI).

Security: API-key auth, per-key rate limiting, PII masking, audit log.
Ops: /health, /ready, Prometheus /metrics, drift report, model registry metadata.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse

from atlas import __version__
from atlas.agent.graph import SupportAgent
from atlas.analytics import load_queries, run_named
from atlas.api.schemas import (
    BatchClassifyRequest,
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    SearchHit,
    SearchRequest,
)
from atlas.config import get_settings
from atlas.db import read_only_query
from atlas.llm.providers import build_llm
from atlas.ml import anomaly
from atlas.ml.service import TriageService
from atlas.monitoring.drift import DriftMonitor
from atlas.monitoring.metrics import METRICS
from atlas.rag.retriever import Retriever
from atlas.security.guardrails import AuditLog
from atlas.security.pii import mask_pii

log = logging.getLogger("atlas.api")


class Container:
    """Long-lived resources, created once at startup."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.triage = TriageService()
        self.retriever = Retriever(settings)
        self.llm = build_llm(settings)
        self.agent = SupportAgent(self.llm, self.retriever, self.triage)
        self.anomaly_detector = self.triage.registry.load("anomaly")
        reference = json.loads((settings.models_dir / "reference_distribution.json").read_text())
        self.drift = DriftMonitor(reference)
        self.audit = AuditLog(settings.artifacts_dir / "audit" / "audit.jsonl")
        self.rate: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = Container()
    log.info("Atlas API ready (llm=%s)", app.state.container.llm.provider)
    yield


app = FastAPI(
    title="Atlas — AI Support Copilot",
    version=__version__,
    description="ML ticket triage, RAG over the knowledge base, a LangGraph agent with safe "
    "text-to-SQL, incident detection and SQL analytics.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    METRICS.observe_latency(path, time.perf_counter() - start)
    METRICS.inc("atlas_requests_total", path=path, status=str(response.status_code))
    response.headers["x-request-id"] = request_id
    return response


def container(request: Request) -> Container:
    return request.app.state.container


def authorize(request: Request, x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    c = container(request)
    if x_api_key != c.settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    window, now = c.rate[x_api_key], time.monotonic()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= c.settings.rate_limit_per_minute:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    window.append(now)
    return x_api_key


# ----------------------------------------------------------------------------- ops
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["ops"])
def ready(c: Container = Depends(container)):
    return {
        "status": "ready",
        "llm_provider": c.llm.provider,
        "llm_model": c.llm.model,
        "embedder": c.retriever.embedder.name,
        "models": c.triage.versions,
    }


@app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
def metrics():
    return METRICS.render()


# ----------------------------------------------------------------------------- ML
@app.post(
    "/v1/tickets/classify",
    response_model=ClassifyResponse,
    tags=["ml"],
    dependencies=[Depends(authorize)],
)
def classify(req: ClassifyRequest, c: Container = Depends(container)):
    masked = mask_pii(req.text)
    t = c.triage.classify(masked.text, req.tier, req.channel)
    c.drift.record("category", t.category)
    c.drift.record("priority", t.priority)
    METRICS.inc("atlas_predictions_total", category=t.category, auto_routed=str(t.auto_routed))
    return ClassifyResponse(**asdict(t), masked_text=masked.text, pii_detected=masked.found)


@app.post(
    "/v1/tickets/classify/batch",
    response_model=list[ClassifyResponse],
    tags=["ml"],
    dependencies=[Depends(authorize)],
)
def classify_batch(req: BatchClassifyRequest, c: Container = Depends(container)):
    masked = [mask_pii(i.text) for i in req.items]
    results = c.triage.classify_many(
        [m.text for m in masked], [i.tier for i in req.items], [i.channel for i in req.items]
    )
    out = []
    for m, t in zip(masked, results, strict=True):
        c.drift.record("category", t.category)
        c.drift.record("priority", t.priority)
        out.append(ClassifyResponse(**asdict(t), masked_text=m.text, pii_detected=m.found))
    return out


@app.get("/v1/incidents", tags=["ml"], dependencies=[Depends(authorize)])
def incidents(days: int = 30, c: Container = Depends(container)):
    """Score the most recent ``days`` of ticket volume for incident-like spikes."""
    wide = anomaly.daily_matrix(read_only_query(anomaly.DAILY_COUNTS_SQL))
    scored = c.anomaly_detector.score(anomaly.build_features(wide)).tail(days)
    return {
        "method": c.anomaly_detector.method,
        "alerts": anomaly.top_anomalies(scored, k=days),
        "days_scored": len(scored),
    }


@app.get("/v1/models", tags=["mlops"], dependencies=[Depends(authorize)])
def models(c: Container = Depends(container)):
    return {
        name: asdict(c.triage.registry.metadata(name))
        for name in ("category", "priority", "anomaly")
    }


@app.get("/v1/monitoring/drift", tags=["mlops"], dependencies=[Depends(authorize)])
def drift(c: Container = Depends(container)):
    return c.drift.report()


# ----------------------------------------------------------------------------- RAG / agent
@app.post(
    "/v1/kb/search", response_model=list[SearchHit], tags=["rag"], dependencies=[Depends(authorize)]
)
def kb_search(req: SearchRequest, c: Container = Depends(container)):
    hits = c.retriever.search(mask_pii(req.query).text, k=req.k, mode=req.mode)
    return [SearchHit(source=h.source, section=h.section, text=h.text, score=h.score) for h in hits]


@app.post("/v1/agent/chat", response_model=ChatResponse, tags=["agent"])
def chat(req: ChatRequest, api_key: str = Depends(authorize), c: Container = Depends(container)):
    state = c.agent.invoke(req.message, customer_id=req.customer_id)
    METRICS.inc(
        "atlas_agent_requests_total",
        intent=state.get("intent", "unknown"),
        blocked=str(state.get("blocked", False)),
    )
    c.audit.write(
        "agent_chat",
        {
            "input": state["question"],  # already PII-masked; stored only as a hash
            "intent": state.get("intent"),
            "blocked": state.get("blocked", False),
            "pii_detected": state.get("pii_found", {}),
            "citations": state.get("citations", []),
            "sql": state.get("sql"),
            "llm": state.get("llm"),
            "customer_id": req.customer_id,
        },
    )
    return ChatResponse(
        answer=state["answer"],
        intent=state.get("intent", "knowledge"),
        citations=state.get("citations", []),
        triage=state.get("triage"),
        sql=state.get("sql"),
        blocked=state.get("blocked", False),
        pii_detected=state.get("pii_found", {}),
        trace=state.get("steps", []),
        llm=state.get("llm"),
    )


# ----------------------------------------------------------------------------- analytics
@app.get("/v1/analytics", tags=["analytics"], dependencies=[Depends(authorize)])
def analytics_catalog():
    return [{"name": q.name, "description": q.description} for q in load_queries().values()]


@app.get("/v1/analytics/{name}", tags=["analytics"], dependencies=[Depends(authorize)])
def analytics(name: str):
    try:
        df = run_named(name)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return json.loads(df.to_json(orient="records", date_format="iso"))
