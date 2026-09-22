"""Descriptive binary group metrics. A gap is evidence to examine, not a legal verdict."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class FairnessAuditor:
    def __init__(self):
        self.metrics = {
            "demographic_parity": "selection-rate difference",
            "equalized_odds": "max TPR/FPR difference",
        }

    def audit(self, y_true, y_pred, sensitive_features=None):
        y, pred = np.asarray(y_true), np.asarray(y_pred)
        if y.ndim != 1 or pred.shape != y.shape or not len(y):
            raise ValueError("Labels and predictions must be nonempty aligned vectors")
        if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
            raise ValueError("This lesson requires binary labels 0 and 1")
        report = {
            "overall_accuracy": float(accuracy_score(y, pred)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "fairness_metrics": {},
            "groups": {},
            "limitations": "Descriptive sample statistics; not proof of fairness or compliance.",
        }
        if sensitive_features is None:
            return report
        groups = np.asarray(sensitive_features)
        if groups.shape != y.shape or any(v is None or str(v) == "nan" for v in groups):
            raise ValueError("Group labels must be present and aligned with observations")
        for group in np.unique(groups):
            mask = groups == group
            truth, guess = y[mask], pred[mask]
            pos, neg = truth == 1, truth == 0
            report["groups"][str(group)] = {
                "count": int(mask.sum()),
                "accuracy": float((truth == guess).mean()),
                "selection_rate": float(guess.mean()),
                "true_positive_rate": float(guess[pos].mean()) if pos.any() else None,
                "false_positive_rate": float(guess[neg].mean()) if neg.any() else None,
            }
        rows = list(report["groups"].values())

        def gap(key):
            vals = [v[key] for v in rows]
            return (
                float(max(vals) - min(vals))
                if len(vals) > 1 and all(v is not None for v in vals)
                else None
            )

        tpr, fpr = gap("true_positive_rate"), gap("false_positive_rate")
        report["fairness_metrics"] = {
            "demographic_parity": gap("selection_rate"),
            "equalized_odds": max(tpr, fpr) if tpr is not None and fpr is not None else None,
        }
        return report
