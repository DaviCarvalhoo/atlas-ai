# Atlas — Design Spec

_Date: 2026-10-01 · Status: approved (approach A)_

## 1. Business problem

A mid-size Brazilian e-commerce company ("AtlasShop") receives thousands of support tickets per
month. Agents lose time **routing** tickets to the right queue, **searching** the policy knowledge
base, and **looking up** customer/order data in several systems. Incidents (e.g. a payment gateway
outage) are noticed only hours later, when the queue explodes.

**Atlas** is a support copilot that:

| Need | Solution | Measurable KPI |
|---|---|---|
| Route tickets automatically | ML classifier (category + priority) | macro-F1, accuracy, % auto-routed above confidence threshold |
| Answer policy questions with sources | RAG over the knowledge base | hit@k, MRR, groundedness |
| Answer "where is my order?" | Agent tool: read-only, whitelisted SQL | routing accuracy, SQL safety tests |
| Detect incidents early | Anomaly detection on daily ticket volume | precision / recall on labelled incident days |
| Protect customer data | PII masking before any LLM call, audit log | 0 PII leaks in tests |

## 2. Architecture

```
                 ┌───────────── FastAPI (auth, rate-limit, /metrics) ─────────────┐
  client ──────► │ /tickets/classify  /kb/search  /agent/chat  /analytics/*        │
                 └──────┬──────────────────┬──────────────────┬───────────────────┘
                        │                  │                  │
                 ML models (sklearn)   RAG retriever      LangGraph agent
                 joblib + MLflow       Chroma + embeddings   guard → route → tool → answer
                        │                  │                  │
                        └──────── SQL (SQLite / Postgres via SQLAlchemy) ───────────┘
```

Modules (`src/atlas/`), each with one responsibility:

- `config` — typed settings from env (`pydantic-settings`).
- `data` — synthetic, seeded dataset generator + SQL schema loader.
- `db` — engine, read-only safe query execution.
- `ml` — text classifier, anomaly detector, training pipeline with MLflow tracking & versioned registry.
- `rag` — chunking, pluggable embeddings (LSA offline / sentence-transformers / OpenAI), Chroma store, retriever.
- `llm` — provider abstraction: OpenAI, Anthropic, Azure OpenAI, Offline (deterministic).
- `agent` — LangGraph state machine and tools.
- `security` — PII masking, prompt-injection heuristics, SQL guard, audit log.
- `monitoring` — request metrics, prediction drift (PSI).
- `eval` — offline evaluation for ML, RAG and agent routing; writes JSON reports.
- `api` — FastAPI app.

## 3. Key decisions

- **Runs with zero keys.** `ATLAS_LLM_PROVIDER=offline` + LSA embeddings make the whole stack
  (tests, CI, Docker demo) reproducible and free. Real providers are a config switch.
- **Synthetic but realistic data**, generated from a fixed seed with label noise and injected
  incident days, so metrics are honest (not 100%).
- **Text-to-SQL is constrained**: SELECT-only, table whitelist, forced LIMIT, single statement,
  executed on a read-only connection. LLM-generated SQL is validated by the same guard.
- **Monolith with clear module boundaries** — simple to run, easy to split later.

## 4. Testing & quality

- Unit tests per module, API tests with `TestClient`, security tests (PII, SQL injection, prompt injection).
- Quality gates in CI: minimum macro-F1 for the classifier and minimum hit@3 for retrieval.

## 5. Out of scope

Frontend UI, real customer data, fine-tuning, multi-tenant auth.
