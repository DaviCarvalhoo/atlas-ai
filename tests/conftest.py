"""Test setup: build an isolated Atlas environment (DB, models, index) in a temp directory once."""

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="atlas-test-"))
os.environ.update(
    {
        "ATLAS_DATABASE_URL": f"sqlite:///{(_TMP / 'atlas.db').as_posix()}",
        "ATLAS_DATA_DIR": str(_TMP / "data"),
        "ATLAS_ARTIFACTS_DIR": str(_TMP / "artifacts"),
        "MLFLOW_TRACKING_URI": f"sqlite:///{(_TMP / 'mlflow.db').as_posix()}",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "ATLAS_LLM_PROVIDER": "offline",
        "ATLAS_EMBEDDING_PROVIDER": "lsa",
        "ATLAS_API_KEY": "test-key",
        "ATLAS_RATE_LIMIT_PER_MINUTE": "1000",
    }
)


@pytest.fixture(scope="session")
def built_env():
    from atlas.data.pipeline import build_database
    from atlas.ml.train import run_training
    from atlas.rag.retriever import build_index

    profile = build_database(seed=42)
    report = run_training()
    index = build_index()
    return {"profile": profile, "training": report, "index": index, "tmp": _TMP}


@pytest.fixture(scope="session")
def agent(built_env):
    from atlas.agent.graph import SupportAgent
    from atlas.llm.providers import build_llm
    from atlas.ml.service import TriageService
    from atlas.rag.retriever import Retriever

    return SupportAgent(build_llm(), Retriever(), TriageService())
