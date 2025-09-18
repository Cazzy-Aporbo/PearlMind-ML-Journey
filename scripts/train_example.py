# scripts/train_example.py
"""First working example: Train XGBoost with fairness audit."""

from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pearlmind.models.ensemble import XGBoostModel
from pearlmind.evaluation import FairnessAuditor
from pearlmind.utils.config import Config

def main():
    # Generate synthetic data with bias
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        flip_y=0.05,
        random_state=42
    )
    
    # Create synthetic sensitive attribute
    sensitive_attr = np.random.choice([0, 1], size=len(y), p=[0.3, 0.7])
    
    # Split data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_attr, test_size=0.2, random_state=42
    )
    
    # Train model
    model = XGBoostModel(n_estimators=100, enable_fairness_audit=True)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Run fairness audit
    audit_report = model.audit_fairness(
        X_test, y_test, sensitive_features=s_test
    )
    
    print(f"Accuracy: {audit_report['overall_accuracy']:.3f}")
    print(f"Fairness Metrics: {audit_report['fairness_metrics']}")
    
if __name__ == "__main__":
    main()
