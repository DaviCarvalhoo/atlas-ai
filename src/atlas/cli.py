"""Command-line entry point: ``atlas <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_build_db(args) -> None:
    from atlas.data.pipeline import build_database

    _print(build_database(seed=args.seed))


def cmd_train(_) -> None:
    from atlas.ml.train import run_training

    r = run_training()
    _print(
        {
            "category": {k: r["category"][k] for k in ("macro_f1", "accuracy", "selected_model")},
            "priority": {k: r["priority"][k] for k in ("macro_f1", "urgent_recall")},
            "anomaly": {
                "champion": r["anomaly"]["champion"],
                **r["anomaly"][r["anomaly"]["champion"]],
            },
        }
    )


def cmd_index(_) -> None:
    from atlas.rag.retriever import build_index

    _print(build_index())


def cmd_evaluate(args) -> None:
    from atlas.eval.run import run_evaluation

    report = run_evaluation(enforce_gates=not args.no_gates)
    _print(report["quality_gates"])


def cmd_bootstrap(args) -> None:
    for step in (cmd_build_db, cmd_train, cmd_index):
        step(args)
    if not args.skip_eval:
        cmd_evaluate(args)


def cmd_ask(args) -> None:
    from atlas.agent.graph import SupportAgent
    from atlas.llm.providers import build_llm
    from atlas.ml.service import TriageService
    from atlas.rag.retriever import Retriever

    agent = SupportAgent(build_llm(), Retriever(), TriageService())
    state = agent.invoke(" ".join(args.question), customer_id=args.customer_id)
    print(state["answer"])
    print("\n— intent:", state.get("intent"), "| citations:", state.get("citations"))
    print("— trace:", *state.get("steps", []), sep="\n   ")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("atlas.api.main:app", host=args.host, port=args.port, reload=False)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="atlas", description="Atlas AI support copilot")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-db", help="generate synthetic data and load it into SQL")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=cmd_build_db)
    sub.add_parser("train", help="train, evaluate and register ML models").set_defaults(
        func=cmd_train
    )
    sub.add_parser("index", help="build the RAG vector index").set_defaults(func=cmd_index)
    e = sub.add_parser("evaluate", help="run the full evaluation suite with quality gates")
    e.add_argument("--no-gates", action="store_true")
    e.set_defaults(func=cmd_evaluate)
    bs = sub.add_parser("bootstrap", help="build-db + train + index + evaluate")
    bs.add_argument("--seed", type=int, default=42)
    bs.add_argument("--skip-eval", action="store_true")
    bs.add_argument("--no-gates", action="store_true")
    bs.set_defaults(func=cmd_bootstrap)
    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question", nargs="+")
    a.add_argument("--customer-id", type=int, default=None)
    a.set_defaults(func=cmd_ask)
    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
