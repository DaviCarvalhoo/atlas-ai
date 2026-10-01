"""Full evaluation: ML (from training report), retrieval, agent routing, answer grounding,
security red-team. Writes ``artifacts/reports/evaluation.json`` and enforces quality gates."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.config import get_settings
from atlas.eval.rag_eval import evaluate_retrieval, load_golden
from atlas.eval.security_eval import evaluate_security
from atlas.llm.providers import normalize_text

AGENT_GOLDEN = Path(__file__).parent / "golden" / "agent.jsonl"

# Minimum acceptable quality. CI fails if any gate is violated.
QUALITY_GATES = {
    "category_macro_f1": 0.85,
    "urgent_recall": 0.75,
    "incident_f1": 0.70,
    "retrieval_hit@3": 0.85,
    "agent_routing_accuracy": 0.85,
    "answer_grounding": 0.60,
    "security_pass_rate": 1.0,
}


def evaluate_agent(agent) -> dict:
    golden = [json.loads(x) for x in AGENT_GOLDEN.read_text(encoding="utf-8").splitlines() if x]
    correct, errors = 0, []
    for item in golden:
        state = agent.invoke(item["question"])
        if state.get("intent") == item["intent"]:
            correct += 1
        else:
            errors.append(
                {
                    "question": item["question"],
                    "expected": item["intent"],
                    "got": state.get("intent"),
                }
            )
    return {"routing_accuracy": round(correct / len(golden), 3), "n": len(golden), "errors": errors}


def evaluate_grounding(agent) -> dict:
    """Does the final answer contain the key fact of the golden answer? (string-match proxy
    for groundedness; an LLM-as-judge can replace it when a provider is configured)."""
    golden = load_golden()
    hits, misses = 0, []
    for item in golden:
        answer = normalize_text(agent.invoke(item["question"])["answer"]).lower()
        expected = [a.lower() for a in [item["answer_contains"], *item.get("aliases", [])]]
        if any(e in answer for e in expected):
            hits += 1
        else:
            misses.append({"question": item["question"], "expected_fact": item["answer_contains"]})
    return {
        "answer_contains_rate": round(hits / len(golden), 3),
        "n": len(golden),
        "misses": misses[:8],
    }


def run_evaluation(enforce_gates: bool = True) -> dict:
    from atlas.agent.graph import SupportAgent
    from atlas.llm.providers import build_llm
    from atlas.ml.service import TriageService
    from atlas.rag.retriever import Retriever

    settings = get_settings()
    training = json.loads((settings.reports_dir / "training_report.json").read_text())
    triage, retriever = TriageService(), Retriever(settings)
    agent = SupportAgent(build_llm(settings), retriever, triage)

    retrieval = evaluate_retrieval(retriever, triage_service=triage)
    agent_eval = evaluate_agent(agent)
    grounding = evaluate_grounding(agent)
    security = evaluate_security()
    anomaly_champion = training["anomaly"]["champion"]

    observed = {
        "category_macro_f1": training["category"]["macro_f1"],
        "urgent_recall": training["priority"]["urgent_recall"],
        "incident_f1": training["anomaly"][anomaly_champion]["f1"],
        "retrieval_hit@3": retrieval["hybrid+triage"]["hit@3"],
        "agent_routing_accuracy": agent_eval["routing_accuracy"],
        "answer_grounding": grounding["answer_contains_rate"],
        "security_pass_rate": round(security["passed"] / security["cases"], 3),
    }
    gates = {
        k: {"observed": observed[k], "minimum": v, "passed": observed[k] >= v}
        for k, v in QUALITY_GATES.items()
    }
    report = {
        "llm_provider": agent.llm.provider,
        "embedder": retriever.embedder.name,
        "quality_gates": gates,
        "ml": {
            "category": {
                k: training["category"][k]
                for k in (
                    "accuracy",
                    "macro_f1",
                    "auto_route_rate",
                    "auto_route_accuracy",
                    "selected_model",
                    "cv_scores",
                )
            },
            "priority": {
                k: training["priority"][k] for k in ("accuracy", "macro_f1", "urgent_recall")
            },
            "anomaly": {
                k: training["anomaly"][k] for k in ("champion", "robust_z", "isolation_forest")
            },
        },
        "retrieval": retrieval,
        "agent": agent_eval,
        "grounding": grounding,
        "security": security,
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    failed = [k for k, g in gates.items() if not g["passed"]]
    if enforce_gates and failed:
        raise SystemExit(f"Quality gates failed: {failed}")
    return report
