import pandas as pd
import pytest

from atlas.data.generate import CATEGORIES, generate
from atlas.data.pipeline import validate
from atlas.ml.registry import ModelRegistry
from atlas.monitoring.drift import DriftMonitor, psi, psi_status


def test_generator_is_deterministic():
    a = generate(seed=7, n_customers=50, n_orders=100, days=20)
    b = generate(seed=7, n_customers=50, n_orders=100, days=20)
    pd.testing.assert_frame_equal(a.tickets, b.tickets)


def test_generated_data_passes_quality_checks():
    ds = generate(seed=1, n_customers=50, n_orders=100, days=30)
    profile = validate(ds.tickets)
    assert profile["rows"] == len(ds.tickets) > 0
    assert set(ds.tickets["category"]) <= set(CATEGORIES)


def test_validation_rejects_bad_data():
    ds = generate(seed=1, n_customers=20, n_orders=40, days=10)
    bad = ds.tickets.copy()
    bad.loc[0, "category"] = "spam"
    with pytest.raises(ValueError):
        validate(bad)


def test_training_meets_quality_bar(built_env):
    report = built_env["training"]
    assert report["category"]["macro_f1"] >= 0.85
    assert report["category"]["selected_model"] != "dummy_most_frequent"
    assert report["priority"]["urgent_recall"] >= 0.75
    champion = report["anomaly"]["champion"]
    assert report["anomaly"][champion]["f1"] >= 0.7


def test_split_is_out_of_time(built_env):
    from atlas.ml.train import load_training_frame, temporal_split

    train, test = temporal_split(load_training_frame())
    assert train["created_at"].max() < test["created_at"].min()


def test_registry_versions_and_promotion(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("m", {"w": 1}, {"f1": 0.5}, {}, "abc")
    mv2 = reg.register("m", {"w": 2}, {"f1": 0.6}, {}, "abc")
    assert reg.versions("m") == [1, 2]
    assert reg.production_version("m") == mv2.version
    reg.promote("m", 1)
    assert reg.load("m") == {"w": 1}


def test_triage_service_predicts(built_env):
    from atlas.ml.service import TriageService

    t = TriageService().classify("Fui cobrado duas vezes no cartão pelo pedido 10234")
    assert t.category == "billing"
    assert 0 <= t.category_confidence <= 1


def test_psi_detects_shift():
    ref = {"a": 0.5, "b": 0.5}
    assert psi(ref, ref) == 0
    assert psi_status(psi(ref, {"a": 0.9, "b": 0.1})) == "significant"


def test_drift_monitor_window():
    mon = DriftMonitor({"category": {"billing": 0.5, "shipping": 0.5}}, min_samples=10)
    assert mon.report()["category"]["status"] == "insufficient_data"
    for _ in range(20):
        mon.record("category", "billing")
    assert mon.report()["category"]["status"] == "significant"
