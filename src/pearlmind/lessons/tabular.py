"""A complete, small experiment with a held-out set and portable outputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pearlmind.models import XGBoostModel
from pearlmind.utils.config import Config


def run(output="outputs/tabular", config=None, data=None, target="target"):
    config = config or Config()
    seed = config.training.seed
    if data:
        frame = pd.read_csv(data)
        if target not in frame:
            raise ValueError(f"Missing target column: {target}")
        y = frame.pop(target).to_numpy()
        groups = frame.pop("group").to_numpy() if "group" in frame else None
        X = frame.to_numpy(dtype=float)
        names = frame.columns.tolist()
    else:
        X, y = make_classification(
            n_samples=600, n_features=8, n_informative=5, n_redundant=1, random_state=seed
        )
        groups = np.random.default_rng(seed).integers(0, 2, len(y))
        names = [f"feature_{i}" for i in range(X.shape[1])]
    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Use finite numeric features and binary target labels 0/1")
    train, test = train_test_split(
        np.arange(len(y)), test_size=config.training.test_size, random_state=seed, stratify=y
    )
    model = XGBoostModel(**config.model.params, random_state=seed)
    model.fit(X[train], y[train])
    report = model.audit_fairness(X[test], y[test], None if groups is None else groups[test])
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "model")
    predictions = pd.DataFrame(
        {
            "row_id": test,
            "actual": y[test],
            "prediction": model.predict(X[test]),
            "probability": model.predict_proba(X[test])[:, 1],
        }
    )
    predictions.to_csv(path / "predictions.csv", index=False)
    report.update(
        {
            "seed": seed,
            "train_rows": len(train),
            "test_rows": len(test),
            "features": names,
            "source": "user CSV" if data else "synthetic",
            "split": "stratified random; use temporal/group splits when observations are dependent",
        }
    )
    (path / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    model.get_feature_importance(feature_names=names).to_csv(path / "importance.csv", index=False)
    return report
