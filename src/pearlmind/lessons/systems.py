"""Two small systems experiments: weighted updates and a bounded tool router.

No model provider, network call or autonomous side effect is required.
"""

import json
from pathlib import Path
import numpy as np


def aggregate_gradients(gradients, counts):
    """Match a full-batch mean when workers report mean gradients at equal weights."""
    g, n = np.asarray(gradients, dtype=float), np.asarray(counts, dtype=float)
    if (
        g.ndim != 2
        or n.shape != (len(g),)
        or not np.isfinite(g).all()
        or not np.isfinite(n).all()
        or (n <= 0).any()
    ):
        raise ValueError("Finite gradient vectors and positive worker sample counts are required")
    return np.average(g, axis=0, weights=n)


def gradient_demo():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    w = np.zeros(2)
    shards = [np.arange(2), np.arange(2, 5)]
    local = [2 * X[s].T @ (X[s] @ w - y[s]) / len(s) for s in shards]
    global_gradient = 2 * X.T @ (X @ w - y) / len(y)
    weighted = aggregate_gradients(local, [2, 3])
    return {
        "worker_counts": [2, 3],
        "worker_gradients": np.array(local).tolist(),
        "weighted_gradient": weighted.tolist(),
        "full_batch_gradient": global_gradient.tolist(),
        "unweighted_gradient": np.mean(local, axis=0).tolist(),
        "matches_full_batch": bool(np.allclose(weighted, global_gradient)),
        "boundary": "Single-process arithmetic simulation, not a distributed training benchmark. Multiple local optimizer steps change the equivalence.",
    }


def route_request(task, approved=False, budget=2):
    """A deterministic teaching state machine: classify → authorize → return or defer."""
    if not isinstance(task, str) or not task.strip() or not isinstance(budget, int) or budget < 0:
        raise ValueError("Provide a nonempty request and nonnegative integer step budget")
    trace = [{"state": "received", "input": task}]
    if budget == 0:
        return {"outcome": "deferred", "reason": "step budget exhausted", "trace": trace}
    if task == "inspect_metrics":
        trace.append({"state": "read_only", "tool": "local_metrics_schema"})
        return {
            "outcome": "completed",
            "result": ["accuracy", "sample_count", "split"],
            "trace": trace,
        }
    if task == "publish_model":
        trace.append({"state": "approval_gate", "approved": approved})
        return {
            "outcome": "ready_for_review" if approved and budget >= 2 else "deferred",
            "reason": "No deployment tool is connected",
            "trace": trace,
        }
    trace.append({"state": "unsupported", "tool": None})
    return {"outcome": "deferred", "reason": "No allowlisted tool for this request", "trace": trace}


def run(output="outputs/systems"):
    report = {
        "gradient_aggregation": gradient_demo(),
        "agent_traces": [
            route_request("inspect_metrics"),
            route_request("publish_model"),
            route_request("delete_database"),
        ],
    }
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
