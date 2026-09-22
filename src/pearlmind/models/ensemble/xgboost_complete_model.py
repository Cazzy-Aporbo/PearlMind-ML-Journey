"""
Alternative scikit-learn-style wrapper retained for the original training script.
Use the core wrapper for the recommended CLI path.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    print("Please install xgboost: pip install xgboost")
    raise


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """
    Standalone teaching XGBoost wrapper; external validation is still required.

    Mathematical Foundation:
        Objective: L(θ) = Σ l(yi, ŷi) + Σ Ω(fk)
        where Ω(f) = γT + ½λ||w||²
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        early_stopping_rounds: Optional[int] = 10,
        random_state: int = 42,
        enable_fairness_audit: bool = True,
        verbosity: int = 1,
        n_jobs: int = -1,
        **kwargs,
    ):
        """Initialize XGBoost model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.enable_fairness_audit = enable_fairness_audit
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        self.model_ = None
        self.is_fitted_ = False
        self.classes_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y, eval_set=None, sample_weight=None, verbose=True):
        """
        Fit the XGBoost model.

        Args:
            X: Training features (array-like)
            y: Training labels
            eval_set: Validation set for early stopping
            sample_weight: Sample weights
            verbose: Print training progress

        Returns:
            self: Fitted model
        """
        # Convert to numpy if needed
        X = np.asarray(X)
        y = np.asarray(y)

        # Handle classification
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        if self.n_classes_ < 2:
            raise ValueError("At least two classes are required")
        if self.n_classes_ > 2:
            self.objective = "multi:softprob"
            self.eval_metric = "mlogloss"

        # Set up parameters
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "verbosity": self.verbosity if verbose else 0,
            "n_jobs": self.n_jobs,
            **self.kwargs,
        }

        # Add num_class for multiclass
        if self.n_classes_ > 2:
            params["num_class"] = self.n_classes_

        if eval_set is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        # Create and train model
        self.model_ = xgb.XGBClassifier(**params)

        # Prepare eval_set if provided
        eval_set_processed = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            y_eval = le.transform(y_eval)
            eval_set_processed = [(X_eval, y_eval)]

        # Fit model
        self.model_.fit(
            X, y_encoded, eval_set=eval_set_processed, sample_weight=sample_weight, verbose=verbose
        )

        self.is_fitted_ = True
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        predictions = self.model_.predict(X).astype(int)

        # Map back to original classes
        return self.classes_[predictions]

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X)
        return self.model_.predict_proba(X)

    def score(self, X, y):
        """Return accuracy score."""
        return accuracy_score(y, self.predict(X))

    def audit_fairness(self, X, y_true, sensitive_features=None, metric_names=None):
        """
        Perform comprehensive fairness audit.

        Args:
            X: Features
            y_true: True labels
            sensitive_features: Protected attributes (array-like)
            metric_names: List of metrics to compute

        Returns:
            Dictionary containing fairness metrics
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before auditing")

        from pearlmind.evaluation import FairnessAuditor

        if self.n_classes_ != 2:
            raise ValueError(
                "The group audit is a binary lesson; multiclass needs a separate metric design"
            )
        y_true = np.asarray(y_true)
        if not np.isin(y_true, self.classes_).all():
            raise ValueError("Observed labels must belong to the trained classes")
        report = FairnessAuditor().audit(
            (y_true == self.classes_[1]).astype(int),
            (self.predict(X) == self.classes_[1]).astype(int),
            sensitive_features,
        )
        # Preserve the original script's report keys while sharing the checked calculations.
        return {
            "overall": {
                "accuracy": report["overall_accuracy"],
                "confusion_matrix": report["confusion_matrix"],
            },
            "fairness": report["fairness_metrics"],
            "by_group": report["groups"],
            "limitations": report["limitations"],
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model_.save_model(path.with_suffix(".model.json"))
        metadata = {
            "classes": self.classes_.tolist(),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load native JSON and explicit metadata; legacy pickle files are not executed."""
        path = Path(path)
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        model = cls(n_estimators=metadata["n_estimators"], max_depth=metadata["max_depth"])
        model.model_ = xgb.XGBClassifier()
        model.model_.load_model(path.with_suffix(".model.json"))
        model.classes_ = np.asarray(metadata["classes"])
        model.n_classes_ = len(model.classes_)
        model.feature_importances_ = model.model_.feature_importances_
        model.is_fitted_ = True
        return model

    def get_feature_importance(self, feature_names=None):
        """Get feature importance with optional names."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted")

        importance = self.feature_importances_

        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return importance


# Standalone test script
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification

    print("Creating sample data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        class_sep=0.8,
        random_state=42,
    )

    # Create synthetic sensitive attribute (with some correlation to outcome)
    sensitive = np.random.choice([0, 1], size=len(y))
    # Introduce some bias
    sensitive[y == 1] = np.random.choice([0, 1], size=(y == 1).sum(), p=[0.3, 0.7])

    # Split data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )

    # Create and train model
    print("\nTraining XGBoost model...")
    model = XGBoostModel(
        n_estimators=50, max_depth=4, learning_rate=0.1, enable_fairness_audit=True
    )

    # Add validation set for early stopping
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_test, y_test, s_test, test_size=0.5, random_state=42
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    # Run fairness audit
    print("\nRunning fairness audit...")
    audit_report = model.audit_fairness(X_test, y_test, sensitive_features=s_test)

    print("\n=== Fairness Audit Report ===")
    print(f"\nOverall Metrics:")
    for key, value in audit_report["overall"].items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.3f}")

    if "by_group" in audit_report:
        print(f"\nMetrics by Group:")
        for group, metrics in audit_report["by_group"].items():
            print(f"\n  {group}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.3f}")
                else:
                    print(f"    {metric}: {value}")

    if "fairness" in audit_report:
        print(f"\nFairness Metrics:")
        for metric, value in audit_report["fairness"].items():
            print(f"  {metric}: {value:.3f}")

    # Test save/load
    print("\n\nTesting save/load...")
    model.save("test_model.pkl")
    loaded_model = XGBoostModel.load("test_model.pkl")

    # Verify loaded model works
    y_pred_loaded = loaded_model.predict(X_test[:5])
    print(f"Predictions from loaded model: {y_pred_loaded}")
    print("\nSuccess! Model is working correctly.")
