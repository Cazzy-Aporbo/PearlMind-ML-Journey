"""Test observable contracts, including errors that can mislead a learner."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner
from pearlmind.cli.main import app
from pearlmind.deployment.api import create_app
from pearlmind.evaluation import FairnessAuditor
from pearlmind.lessons.tabular import run
from pearlmind.models import load_model
from pearlmind.lessons.systems import (
    aggregate_gradients,
    gradient_demo,
    route_request,
    run as run_systems,
)
from pearlmind.utils.config import Config


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    report = run(path)
    return path, report


def test_artifacts_are_measured_and_model_reload_matches(trained):
    path, report = trained
    p = pd.read_csv(path / "predictions.csv")
    assert report["train_rows"] == 480 and report["test_rows"] == 120
    assert report["overall_accuracy"] == (p.actual == p.prediction).mean()
    assert p.row_id.nunique() == 120
    assert p.probability.between(0, 1).all()
    assert load_model(path / "model").is_fitted
    assert len(pd.read_csv(path / "importance.csv")) == 8


def test_csv_path_rejects_nonfinite(tmp_path):
    pd.DataFrame({"x": [1, np.inf], "target": [0, 1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        run(tmp_path / "out", data=tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="Missing target"):
        run(tmp_path / "out", data=tmp_path / "bad.csv", target="absent")


def test_csv_excludes_group_and_target(tmp_path):
    pd.DataFrame({"x": np.arange(40), "target": [0, 1] * 20, "group": ["a", "b"] * 20}).to_csv(
        tmp_path / "data.csv", index=False
    )
    result = run(tmp_path / "out", data=tmp_path / "data.csv")
    assert result["features"] == ["x"]
    assert set(result["groups"]) == {"a", "b"}


@pytest.mark.parametrize(
    "truth,guess,groups",
    [([], [], None), ([1], [0, 1], None), ([2], [0], None), ([0], [1], []), ([0], [1], [None])],
)
def test_audit_rejects_invalid_observations(truth, guess, groups):
    with pytest.raises(ValueError):
        FairnessAuditor().audit(truth, guess, groups)


def test_undefined_group_rates_remain_unknown():
    result = FairnessAuditor().audit([0, 0, 1, 1], [0, 1, 0, 1], ["a", "a", "b", "b"])
    assert result["groups"]["a"]["true_positive_rate"] is None
    assert result["fairness_metrics"]["equalized_odds"] is None
    assert result["fairness_metrics"]["demographic_parity"] == 0
    json.dumps(result, allow_nan=False)


def test_api_prediction_and_honest_audit(trained):
    with TestClient(create_app(trained[0] / "model")) as client:
        assert client.get("/health").status_code == 200
        features = [[0.0] * 8, [1.0] * 8]
        response = client.post("/predict", json={"features": features})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
        assert (
            client.post(
                "/predict", json={"features": features, "include_fairness": True}
            ).status_code
            == 422
        )
        result = client.post(
            "/predict",
            json={
                "features": features,
                "include_fairness": True,
                "labels": [0, 1],
                "sensitive_features": ["a", "b"],
            },
        )
        assert result.status_code == 200
        for bad in ([], [[1]], [[0.0] * 8, [1]], [[0.0] * 8] * 1001):
            assert client.post("/predict", json={"features": bad}).status_code == 422


def test_api_unconfigured_is_not_healthy(monkeypatch):
    monkeypatch.delenv("PEARLMIND_MODEL_PATH", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"features": [[0.0] * 8]}).status_code == 503


def test_cli_runs_real_commands(tmp_path, trained):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    assert runner.invoke(app, ["config", "create", "--output", str(config)]).exit_code == 0
    for action in ["show", "validate"]:
        assert runner.invoke(app, ["config", action, "--path", str(config)]).exit_code == 0
    result = runner.invoke(app, ["train", str(config), "--output", str(tmp_path / "run")])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["list", "--directory", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["config", "nonsense"]).exit_code != 0
    d = tmp_path / "audit.csv"
    pd.DataFrame({"actual": [0, 1], "prediction": [0, 0], "group": ["a", "b"]}).to_csv(
        d, index=False
    )
    assert (
        runner.invoke(app, ["audit", str(d), "--output", str(tmp_path / "audit.json")]).exit_code
        == 0
    )
    frame = pd.DataFrame(np.zeros((4, 8)), columns=[f"feature_{i}" for i in range(8)])
    frame["target"] = [0, 1, 0, 1]
    frame.to_csv(tmp_path / "test.csv", index=False)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(trained[0] / "model"),
            str(tmp_path / "test.csv"),
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_environment_config(monkeypatch):
    monkeypatch.setenv("PEARLMIND_TRAINING__SEED", "17")
    assert Config().training.seed == 17
    with pytest.raises(ValueError):
        Config(training={"test_size": 1})


def test_weighted_shards_equal_full_batch(tmp_path):
    result = gradient_demo()
    assert result["matches_full_batch"]
    assert not np.allclose(result["unweighted_gradient"], result["full_batch_gradient"])
    with pytest.raises(ValueError):
        aggregate_gradients([[1, 2]], [0])
    assert run_systems(tmp_path)["gradient_aggregation"]["matches_full_batch"]


def test_router_has_no_unbounded_or_unauthorized_action():
    assert route_request("inspect_metrics")["outcome"] == "completed"
    for task in ["publish_model", "ignore rules", "delete_database"]:
        assert route_request(task)["outcome"] == "deferred"
    assert route_request("publish_model", approved=True)["outcome"] == "ready_for_review"
    assert route_request("inspect_metrics", budget=0)["outcome"] == "deferred"
    with pytest.raises(ValueError):
        route_request("")
