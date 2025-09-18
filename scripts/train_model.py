"""
Complete training script with fairness audit.
Save as: scripts/train_model.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our model
from pearlmind.models.ensemble.xgboost_model import XGBoostModel


def create_biased_dataset(n_samples=5000, bias_strength=0.7):
    """
    Create a synthetic dataset with intentional bias for testing fairness.
    
    This simulates a scenario where a protected attribute (e.g., gender/race)
    influences the outcome in a biased way.
    """
    print("Creating synthetic biased dataset...")
    
    # Create base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=42
    )
    
    # Create sensitive attribute (0 = unprivileged, 1 = privileged)
    sensitive = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Introduce bias: privileged group more likely to get positive outcome
    bias_mask = np.random.random(n_samples) < bias_strength
    for i in range(n_samples):
        if bias_mask[i]:
            if sensitive[i] == 1 and y[i] == 0:
                # Flip some negatives to positive for privileged group
                if np.random.random() < 0.3:
                    y[i] = 1
            elif sensitive[i] == 0 and y[i] == 1:
                # Flip some positives to negative for unprivileged group
                if np.random.random() < 0.3:
                    y[i] = 0
                    
    return X, y, sensitive


def plot_fairness_results(audit_report, save_path=None):
    """Create visualizations of fairness metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Fairness Audit Results', fontsize=16)
    
    # Plot 1: Accuracy by group
    if "by_group" in audit_report:
        groups = list(audit_report["by_group"].keys())
        accuracies = [audit_report["by_group"][g]["accuracy"] for g in groups]
        
        ax = axes[0, 0]
        bars = ax.bar(groups, accuracies, color=['#FFCFE7', '#F6EAFE'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Group')
        ax.set_ylim([0, 1])
        ax.axhline(y=audit_report["overall"]["accuracy"], color='#6B5B95', 
                   linestyle='--', label='Overall')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        ax.legend()
    
    # Plot 2: Positive rates by group (Demographic Parity)
    if "by_group" in audit_report:
        positive_rates = [audit_report["by_group"][g]["positive_rate"] for g in groups]
        
        ax = axes[0, 1]
        bars = ax.bar(groups, positive_rates, color=['#A8E6CF', '#FFE4F1'])
        ax.set_ylabel('Positive Prediction Rate')
        ax.set_title('Demographic Parity Check')
        ax.set_ylim([0, 1])
        
        for bar, rate in zip(bars, positive_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
    
    # Plot 3: Confusion matrices by group
    if "by_group" in audit_report:
        ax = axes[1, 0]
        tpr_values = []
        fpr_values = []
        
        for group in groups:
            metrics = audit_report["by_group"][group]
            if "true_positive_rate" in metrics:
                tpr_values.append(metrics["true_positive_rate"])
                fpr_values.append(metrics["false_positive_rate"])
        
        if tpr_values:
            x = np.arange(len(groups))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, tpr_values, width, label='TPR', color='#E8D5FF')
            bars2 = ax.bar(x + width/2, fpr_values, width, label='FPR', color='#FFCFE7')
            
            ax.set_ylabel('Rate')
            ax.set_title('True Positive vs False Positive Rates')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()
            ax.set_ylim([0, 1])
    
    # Plot 4: Fairness metrics summary
    if "fairness" in audit_report:
        ax = axes[1, 1]
        metrics = audit_report["fairness"]
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Truncate long names
        metric_names = [name.replace('_', ' ').title()[:20] for name in metric_names]
        
        bars = ax.barh(metric_names, metric_values, color='#6B5B95')
        ax.set_xlabel('Value')
        ax.set_title('Fairness Metrics Summary')
        
        for bar, val in zip(bars, metric_values):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{val:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Fairness plots saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train PearlMind model with fairness audit")
    parser.add_argument("--dataset", choices=["synthetic", "breast_cancer"], 
                       default="synthetic", help="Dataset to use")
    parser.add_argument("--n_samples", type=int, default=5000, 
                       help="Number of samples for synthetic data")
    parser.add_argument("--bias_strength", type=float, default=0.7, 
                       help="Bias strength (0-1) for synthetic data")
    parser.add_argument("--save_model", type=str, default="models/xgboost_model.pkl",
                       help="Path to save trained model")
    parser.add_argument("--save_plots", type=str, default="outputs/fairness_audit.png",
                       help="Path to save fairness plots")
    args = parser.parse_args()
    
    # Load or create dataset
    if args.dataset == "synthetic":
        X, y, sensitive = create_biased_dataset(
            n_samples=args.n_samples,
            bias_strength=args.bias_strength
        )
        print(f"Created synthetic dataset with {args.n_samples} samples")
        print(f"Bias strength: {args.bias_strength}")
    else:
        # Use breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        # Create synthetic sensitive attribute for demonstration
        sensitive = np.random.choice([0, 1], size=len(y), p=[0.4, 0.6])
        print("Loaded breast cancer dataset")
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Sensitive attribute distribution: {np.bincount(sensitive)}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split test into validation and test
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_test, y_test, s_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    print(f"Train size: {X_train.shape[0]}")
    print(f"Validation size: {X_val.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\n" + "="*50)
    print("Training XGBoost Model with Fairness Auditing")
    print("="*50)
    
    model = XGBoostModel(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        enable_fairness_audit=True,
        random_state=42
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    test_accuracy = model.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Feature importance
    print("\nTop 5 Most Important Features:")
    importance = model.get_feature_importance()
    top_indices = np.argsort(importance)[-5:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {importance[idx]:.4f}")
    
    # Fairness Audit
    print("\n" + "="*50)
    print("Fairness Audit Report")
    print("="*50)
    
    audit_report = model.audit_fairness(
        X_test, y_test, 
        sensitive_features=s_test
    )
    
    # Print overall metrics
    print("\nOverall Performance:")
    print(f"  Accuracy: {audit_report['overall']['accuracy']:.4f}")
    print(f"  Precision: {audit_report['overall']['precision']:.4f}")
    print(f"  Recall: {audit_report['overall']['recall']:.4f}")
    print(f"  F1 Score: {audit_report['overall']['f1_score']:.4f}")
    if 'auc' in audit_report['overall']:
        print(f"  AUC: {audit_report['overall']['auc']:.4f}")
    
    # Print group metrics
    if "by_group" in audit_report:
        print("\nPerformance by Sensitive Group:")
        for group_name, metrics in audit_report["by_group"].items():
            print(f"\n  {group_name} (n={metrics['size']}):")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Positive Rate: {metrics['positive_rate']:.4f}")
            if 'true_positive_rate' in metrics:
                print(f"    True Positive Rate: {metrics['true_positive_rate']:.4f}")
                print(f"    False Positive Rate: {metrics['false_positive_rate']:.4f}")
    
    # Print fairness metrics
    if "fairness" in audit_report:
        print("\nFairness Metrics:")
        for metric_name, value in audit_report["fairness"].items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Interpret results
        print("\nFairness Interpretation:")
        dp_diff = audit_report["fairness"].get("demographic_parity_diff", 0)
        if dp_diff < 0.1:
            print("  ✓ Good demographic parity (difference < 0.1)")
        else:
            print(f"  ⚠ Demographic parity difference is {dp_diff:.3f} (> 0.1)")
        
        eo_diff = audit_report["fairness"].get("equal_opportunity_diff", 0)
        if eo_diff < 0.1:
            print("  ✓ Good equal opportunity (difference < 0.1)")
        else:
            print(f"  ⚠ Equal opportunity difference is {eo_diff:.3f} (> 0.1)")
    
    # Create visualizations
    print("\nCreating fairness visualizations...")
    Path(args.save_plots).parent.mkdir(parents=True, exist_ok=True)
    plot_fairness_results(audit_report, save_path=args.save_plots)
    
    # Save model
    print(f"\nSaving model to {args.save_model}...")
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_model)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Model saved to: {args.save_model}")
    print(f"Fairness plots saved to: {args.save_plots}")
    
    return model, audit_report


if __name__ == "__main__":
    model, report = main()
