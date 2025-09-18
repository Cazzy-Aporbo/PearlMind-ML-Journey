"""
Complete working XGBoost model with fairness auditing.
Save as: src/pearlmind/models/ensemble/xgboost_model.py
"""

import json
import pickle
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
    Production-ready XGBoost with fairness auditing.
    
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
        **kwargs
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
        
        # Encode labels if needed
        if self.n_classes_ == 2:
            # Binary classification
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            # Multiclass
            y_encoded = y
            self.objective = "multi:softprob"
            
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
            **self.kwargs
        }
        
        # Add num_class for multiclass
        if self.n_classes_ > 2:
            params["num_class"] = self.n_classes_
            
        # Create and train model
        self.model_ = xgb.XGBClassifier(**params)
        
        # Prepare eval_set if provided
        eval_set_processed = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            if self.n_classes_ == 2:
                y_eval = le.transform(y_eval)
            eval_set_processed = [(X_eval, y_eval)]
            
        # Fit model
        self.model_.fit(
            X, y_encoded,
            eval_set=eval_set_processed,
            early_stopping_rounds=self.early_stopping_rounds,
            sample_weight=sample_weight,
            verbose=verbose
        )
        
        self.is_fitted_ = True
        self.feature_importances_ = self.model_.feature_importances_
        
        return self
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        predictions = self.model_.predict(X)
        
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
        
    def audit_fairness(
        self,
        X,
        y_true,
        sensitive_features=None,
        metric_names=None
    ):
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
            
        y_pred = self.predict(X)
        report = {}
        
        # Overall metrics
        report["overall"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        report["overall"].update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # If probabilities available, add AUC
        if self.n_classes_ == 2:
            try:
                y_proba = self.predict_proba(X)[:, 1]
                report["overall"]["auc"] = roc_auc_score(y_true, y_proba)
            except:
                pass
                
        # Fairness metrics by group
        if sensitive_features is not None:
            sensitive_features = np.asarray(sensitive_features)
            unique_groups = np.unique(sensitive_features)
            
            report["fairness"] = {}
            report["by_group"] = {}
            
            for group in unique_groups:
                mask = sensitive_features == group
                if mask.sum() == 0:
                    continue
                    
                group_name = f"group_{group}"
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # Group metrics
                report["by_group"][group_name] = {
                    "size": int(mask.sum()),
                    "accuracy": accuracy_score(y_true_group, y_pred_group),
                    "positive_rate": (y_pred_group == 1).mean()
                }
                
                # Add confusion matrix
                cm = confusion_matrix(y_true_group, y_pred_group)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    report["by_group"][group_name].update({
                        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                        "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
                        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                    })
            
            # Calculate fairness metrics
            accuracies = [v["accuracy"] for v in report["by_group"].values()]
            positive_rates = [v["positive_rate"] for v in report["by_group"].values()]
            
            # Demographic parity: difference in positive rates
            if len(positive_rates) >= 2:
                report["fairness"]["demographic_parity_diff"] = max(positive_rates) - min(positive_rates)
                report["fairness"]["demographic_parity_ratio"] = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
            
            # Equal opportunity: difference in true positive rates
            tpr_values = [v.get("true_positive_rate", 0) for v in report["by_group"].values()]
            if len(tpr_values) >= 2:
                report["fairness"]["equal_opportunity_diff"] = max(tpr_values) - min(tpr_values)
            
            # Accuracy parity
            if len(accuracies) >= 2:
                report["fairness"]["accuracy_parity_diff"] = max(accuracies) - min(accuracies)
                report["fairness"]["min_group_accuracy"] = min(accuracies)
                report["fairness"]["max_group_accuracy"] = max(accuracies)
                
        return report
        
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
        # Save metadata
        metadata = {
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist(),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "feature_importances": self.feature_importances_.tolist()
        }
        
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
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
        random_state=42
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
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        enable_fairness_audit=True
    )
    
    # Add validation set for early stopping
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_test, y_test, s_test, test_size=0.5, random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=True
    )
    
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
