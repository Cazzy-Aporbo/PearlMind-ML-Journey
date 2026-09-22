import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import make_classification
from pearlmind.models.ensemble.xgboost_complete_model import XGBoostModel


def test_original_linear_lesson_accepts_one_row(capsys):
    path = Path(__file__).parents[1] / "Learning/01_linear_regression.py"
    spec = importlib.util.spec_from_file_location("linear_lesson", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GentleLinearRegression(iterations=100, verbose=False)
    X = np.arange(10.0).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    model.fit(X, y)
    assert np.isfinite(model.explain_prediction(X[:1]))
    with pytest.raises(ValueError):
        model.explain_prediction([1, 2])


@pytest.mark.parametrize("classes", [2, 3])
def test_alternate_wrapper_labels_and_json_roundtrip(tmp_path, classes):
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=classes,
        random_state=42,
    )
    labels = np.array([f"class_{i}" for i in y])
    model = XGBoostModel(n_estimators=8, n_jobs=1, max_depth=2)
    with pytest.raises(ValueError):
        model.predict(X)
    model.fit(X[:80], labels[:80], eval_set=(X[80:], labels[80:]), verbose=False)
    pred = model.predict(X[80:])
    assert set(pred) <= set(labels)
    assert model.predict_proba(X[80:]).shape == (20, classes)
    model.save(tmp_path / "model")
    loaded = XGBoostModel.load(tmp_path / "model")
    assert np.array_equal(pred, loaded.predict(X[80:]))
    assert model.score(X[80:], labels[80:]) >= 0
    assert len(model.get_feature_importance()) == 5
    if classes == 2:
        report = model.audit_fairness(X[80:], labels[80:], ["a", "b"] * 10)
        assert report["overall"]["accuracy"] == (pred == labels[80:]).mean()
    else:
        with pytest.raises(ValueError):
            model.audit_fairness(X[80:], labels[80:])
