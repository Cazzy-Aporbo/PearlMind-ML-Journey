"""Compare fixed models on identical held-out rows, retaining paired uncertainty."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def paired_brier_interval(y, candidate, reference, resamples=2000, seed=42):
    """Percentile interval for candidate minus reference squared probability error.

    Rows are resampled together: each outcome stays paired with both predictions.
    This estimates test-row uncertainty conditional on the fitted models, not
    uncertainty from retraining, model selection or a changing population.
    """
    y, candidate, reference = [np.asarray(a, dtype=float) for a in (y, candidate, reference)]
    if (
        y.ndim != 1
        or y.size < 2
        or candidate.shape != y.shape
        or reference.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not all(
            np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in (candidate, reference)
        )
    ):
        raise ValueError(
            "Use aligned one-dimensional binary outcomes and finite probabilities in [0, 1]"
        )
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise ValueError("Use at least 100 integer resamples")
    differences = (candidate - y) ** 2 - (reference - y) ** 2
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    # O(n) temporary memory per resample; never allocate a resamples × rows matrix.
    for i in range(resamples):
        means[i] = differences[rng.integers(0, len(y), len(y))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "lower": float(low),
        "upper": float(high),
        "confidence_level": 0.95,
        "resamples": resamples,
        "method": "paired row percentile bootstrap; conditional on fitted models",
    }


def run(output="outputs/comparison", seed=42, resamples=2000):
    X, y = make_moons(n_samples=600, noise=0.25, random_state=seed)
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
    models = {
        "prior": DummyClassifier(strategy="prior"),
        "linear": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        ),
        "tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=seed),
    }
    metrics, probabilities = {}, {}
    table = pd.DataFrame({"row_id": test, "actual": y[test]})
    for name, model in models.items():
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        probabilities[name] = p
        table[name + "_probability"] = p
        metrics[name] = {
            "accuracy": float(accuracy_score(y[test], model.predict(X[test]))),
            "brier": float(np.mean((p - y[test]) ** 2)),
            "log_loss": float(log_loss(y[test], p, labels=[0, 1])),
        }
    report = {
        "seed": seed,
        "source": "synthetic noisy two-moons",
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": metrics,
        "tree_minus_linear": paired_brier_interval(
            y[test], probabilities["tree"], probabilities["linear"], resamples, seed
        ),
        "design": "Fixed configurations; one held-out split; no test-set tuning",
        "limitations": "Independent row assumption; no retraining uncertainty, deployment cost measurement or population generalization",
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/comparison")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
