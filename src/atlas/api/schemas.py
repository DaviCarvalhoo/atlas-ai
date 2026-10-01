"""Request/response contracts of the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        examples=["Fui cobrado duas vezes no pedido 10234, é urgente!"],
    )
    tier: Literal["standard", "silver", "gold"] = "standard"
    channel: Literal["email", "chat", "whatsapp", "phone"] = "chat"


class ClassifyResponse(BaseModel):
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    auto_routed: bool = Field(description="False → send to human triage (low confidence)")
    masked_text: str
    pii_detected: dict[str, int]
    model_versions: dict[str, int | None]


class BatchClassifyRequest(BaseModel):
    items: list[ClassifyRequest] = Field(..., min_length=1, max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, examples=["prazo para devolução"])
    k: int = Field(4, ge=1, le=10)
    mode: Literal["dense", "keyword", "hybrid"] = "hybrid"


class SearchHit(BaseModel):
    source: str
    section: str
    text: str
    score: float


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=2, max_length=4000, examples=["Meu pedido 10234 está atrasado, o que faço?"]
    )
    customer_id: int | None = Field(
        None, description="When set, order lookups are restricted to this customer's orders"
    )


class ChatResponse(BaseModel):
    answer: str
    intent: str
    citations: list[str]
    triage: dict | None = None
    sql: str | None = None
    blocked: bool = False
    pii_detected: dict[str, int] = {}
    trace: list[str]
    llm: dict | None = None
