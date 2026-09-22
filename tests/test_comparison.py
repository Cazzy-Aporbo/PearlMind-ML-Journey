"""Analytic counterexamples and independent recomputation for the comparison lesson."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from pearlmind.lessons.comparison import paired_brier_interval, run


def test_identical_predictions_have_exact_zero_difference():
    result = paired_brier_interval([0, 1, 0], [0.2, 0.8, 0.6], [0.2, 0.8, 0.6], 100)
    assert result["mean_difference"] == result["lower"] == result["upper"] == 0


def test_perfect_against_inverted_is_negative_one():
    result = paired_brier_interval([0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], 100)
    assert result["mean_difference"] == result["lower"] == result["upper"] == -1


@pytest.mark.parametrize(
    "y,p,q,n",
    [
        ([0, 1], [0.2], [0.3, 0.7], 100),
        ([0, 1], [float("nan"), 0.9], [0.2, 0.8], 100),
        ([0, 1], [0.2, 1.1], [0.2, 0.8], 100),
        ([0, 2], [0.2, 0.9], [0.2, 0.8], 100),
        ([0, 1], [0.2, 0.9], [0.2, 0.8], 2),
    ],
)
def test_invalid_evaluation_contracts(y, p, q, n):
    with pytest.raises(ValueError):
        paired_brier_interval(y, p, q, n)


def test_saved_rows_reproduce_report_and_seed(tmp_path):
    report = run(tmp_path, resamples=100)
    rows = pd.read_csv(tmp_path / "predictions.csv")
    assert len(rows) == report["test_rows"] == 150
    assert rows.row_id.is_unique
    for name in ["prior", "linear", "tree"]:
        assert report["metrics"][name]["brier"] == pytest.approx(
            brier_score_loss(rows.actual, rows[name + "_probability"])
        )
    difference = report["metrics"]["tree"]["brier"] - report["metrics"]["linear"]["brier"]
    assert report["tree_minus_linear"]["mean_difference"] == pytest.approx(difference)
    assert run(tmp_path / "again", resamples=100) == report
