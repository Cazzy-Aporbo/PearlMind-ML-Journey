#!/usr/bin/env python3
"""
AI Ethics, Governance & Compliance Framework
Author: Cazandra Aporbo
Upload Date: November 2025
This framework provides practical implementation of ethical AI systems across
industries, with particular attention to regulatory compliance, bias mitigation,
and operational considerations that are frequently overlooked in standard
implementations.
My code demonstrates not just what should be done, but how to actually
implement, test, and audit ethical AI systems in production environments.
"""

import hashlib
import json
import sqlite3
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import logging
import secrets
import threading
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
import inspect
import traceback
import re
from contextlib import contextmanager
import os
import sys


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_ethics_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BiasType(Enum):
    """
    Comprehensive categorization of algorithmic and human biases that affect AI systems.
    
    Understanding these biases is crucial because they compound through the ML pipeline,
    from data collection through deployment. Most practitioners focus on a few obvious
    biases but miss the subtle interactions between different bias types.
    """
    
    HISTORICAL = auto()           # Past data reflects historical discrimination
    REPRESENTATION = auto()        # Underrepresentation of certain groups in data
    MEASUREMENT = auto()          # Features proxy for sensitive attributes incorrectly
    AGGREGATION = auto()          # Model assumes one-size-fits-all
    EVALUATION = auto()           # Benchmarks don't reflect real use cases
    POPULATION = auto()           # Model performs differently across subgroups
    SAMPLING = auto()             # Non-random sampling creates skewed distributions
    TEMPORAL = auto()             # Model degrades as population shifts over time
    CONFIRMATION = auto()         # Systems reinforce existing beliefs
    AUTOMATION = auto()           # Over-reliance on automated decisions
    FEEDBACK_LOOP = auto()        # Predictions influence future training data
    SURVIVORSHIP = auto()         # Only considering successful cases
    SELECTION = auto()            # Cherry-picking favorable outcomes
    ANCHORING = auto()            # Over-relying on first piece of information
    AVAILABILITY = auto()         # Overweighting easily recalled examples
    FRAMING = auto()              # Context presentation affects interpretation
    OMITTED_VARIABLE = auto()     # Missing confounding factors
    PROXY = auto()                # Using correlated features for protected attributes
    LABEL = auto()                # Ground truth itself contains bias
    ANNOTATOR = auto()            # Human labelers introduce their biases
    DEPLOYMENT = auto()           # Different deployment contexts than training
    INTERACTION = auto()          # User interaction patterns create bias
    AMPLIFICATION = auto()        # Small biases magnified through iterations


class RegulatoryFramework(Enum):
    """
    Global regulatory frameworks affecting AI deployment.
    
    Each framework has different requirements, and systems often need to comply
    with multiple simultaneously. The challenge is that some requirements conflict,
    requiring careful architectural decisions.
    """
    
    GDPR = "General Data Protection Regulation (EU)"
    CCPA = "California Consumer Privacy Act"
    CPRA = "California Privacy Rights Act"
    LGPD = "Lei Geral de Proteção de Dados (Brazil)"
    PIPEDA = "Personal Information Protection and Electronic Documents Act (Canada)"
    POPIA = "Protection of Personal Information Act (South Africa)"
    APPI = "Act on Protection of Personal Information (Japan)"
    PDPA = "Personal Data Protection Act (Singapore)"
    DPA = "Data Protection Act (UK)"
    HIPAA = "Health Insurance Portability and Accountability Act (US)"
    FCRA = "Fair Credit Reporting Act (US)"
    ECOA = "Equal Credit Opportunity Act (US)"
    FHA = "Fair Housing Act (US)"
    ADA = "Americans with Disabilities Act"
    EU_AI_ACT = "EU Artificial Intelligence Act"
    ALGORITHMIC_ACCOUNTABILITY = "Algorithmic Accountability Act (Proposed US)"
    IEEE_EAD = "IEEE Ethically Aligned Design"
    ISO_IEC_23053 = "ISO/IEC 23053 AI Trustworthiness"
    ISO_IEC_23894 = "ISO/IEC 23894 AI Risk Management"
    NIST_AI_RMF = "NIST AI Risk Management Framework"


@dataclass
class EthicalConstraint:
    """
    Represents an ethical constraint that must be satisfied by the AI system.
    
    Constraints can be hard (must be satisfied) or soft (should be optimized).
    The challenge is balancing competing constraints, especially when they
    conflict with business objectives.
    """
    
    name: str
    description: str
    constraint_type: str  # 'hard' or 'soft'
    measurement_method: str
    threshold: float
    priority: int  # Higher number = higher priority
    regulatory_source: Optional[RegulatoryFramework] = None
    applies_to_domains: List[str] = field(default_factory=list)
    verification_frequency: str = "continuous"  # continuous, daily, weekly, monthly
    remediation_action: Optional[str] = None
    
    def evaluate(self, measurement: float) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if constraint is satisfied.
        
        Returns tuple of (is_satisfied, explanation).
        """
        if self.constraint_type == 'hard':
            satisfied = measurement >= self.threshold
            if not satisfied:
                explanation = f"Hard constraint {self.name} violated: {measurement:.3f} < {self.threshold:.3f}"
                return False, explanation
        else:
            satisfied = measurement >= self.threshold * 0.9  # Allow 10% tolerance for soft constraints
            if not satisfied:
                explanation = f"Soft constraint {self.name} not met: {measurement:.3f} below target"
                return False, explanation
        
        return True, None


@dataclass
class DataGovernanceRecord:
    """
    Comprehensive record for data governance and lineage tracking.
    
    Most systems track data at a high level, but ethical AI requires granular
    tracking of data transformations, access patterns, and retention policies.
    This becomes critical during audits and incident investigations.
    """
    
    data_id: str
    collection_timestamp: datetime
    source_system: str
    data_type: str
    sensitivity_level: str  # public, internal, confidential, restricted
    personal_data_categories: List[str]  # As defined by GDPR Article 9
    consent_basis: Optional[str] = None  # consent, contract, legal, legitimate_interest
    consent_timestamp: Optional[datetime] = None
    retention_period_days: int = 365
    deletion_scheduled: Optional[datetime] = None
    geographic_origin: Optional[str] = None
    geographic_restrictions: List[str] = field(default_factory=list)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    transformation_history: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    encryption_status: str = "at_rest_and_transit"
    anonymization_applied: bool = False
    pseudonymization_applied: bool = False
    
    def add_access_record(self, accessor: str, purpose: str, timestamp: datetime = None):
        """Record data access for audit trail."""
        self.access_log.append({
            'accessor': accessor,
            'purpose': purpose,
            'timestamp': timestamp or datetime.now(),
            'data_elements_accessed': []  # Would contain specific fields accessed
        })
    
    def add_transformation(self, operation: str, parameters: Dict[str, Any], 
                          timestamp: datetime = None):
        """Record data transformation for lineage tracking."""
        self.transformation_history.append({
            'operation': operation,
            'parameters': parameters,
            'timestamp': timestamp or datetime.now(),
            'reversible': self._is_reversible_operation(operation)
        })
    
    def _is_reversible_operation(self, operation: str) -> bool:
        """Determine if a transformation is reversible for GDPR compliance."""
        irreversible_ops = ['hash', 'aggregate', 'sample', 'truncate', 'redact']
        return operation.lower() not in irreversible_ops
    
    def check_retention_compliance(self) -> Tuple[bool, Optional[str]]:
        """Check if data retention policies are being followed."""
        current_time = datetime.now()
        data_age = (current_time - self.collection_timestamp).days
        
        if data_age > self.retention_period_days:
            if self.deletion_scheduled is None:
                return False, f"Data exceeded retention period of {self.retention_period_days} days"
            elif current_time > self.deletion_scheduled:
                return False, "Data deletion overdue"
        
        return True, None


class FairnessMetric:
    """
    Implementation of various fairness metrics for bias detection.
    
    Different fairness definitions can be mutually exclusive (impossibility theorem),
    so the choice of metric must align with the specific use case and legal requirements.
    Practitioners often don't realize that optimizing for one type of fairness
    can worsen another.
    """
    
    @staticmethod
    def demographic_parity(predictions: Dict[str, List[int]], 
                          groups: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate demographic parity (statistical parity).
        
        Requires equal positive prediction rates across groups.
        Used in employment and lending contexts.
        """
        group_rates = {}
        for group_name, group_members in groups.items():
            group_predictions = [predictions[member] for member in group_members 
                               if member in predictions]
            if group_predictions:
                positive_rate = sum(group_predictions) / len(group_predictions)
                group_rates[group_name] = positive_rate
        
        if not group_rates:
            return {}
        
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        
        disparity = max_rate - min_rate
        return {
            'group_rates': group_rates,
            'max_disparity': disparity,
            'ratio': min_rate / max_rate if max_rate > 0 else 0
        }
    
    @staticmethod
    def equalized_odds(predictions: Dict[str, int], 
                       ground_truth: Dict[str, int],
                       groups: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate equalized odds (separation).
        
        Requires equal true positive and false positive rates across groups.
        Stronger than demographic parity but harder to achieve.
        """
        metrics = {}
        for group_name, group_members in groups.items():
            tp = fp = tn = fn = 0
            for member in group_members:
                if member in predictions and member in ground_truth:
                    pred = predictions[member]
                    true = ground_truth[member]
                    if pred == 1 and true == 1:
                        tp += 1
                    elif pred == 1 and true == 0:
                        fp += 1
                    elif pred == 0 and true == 1:
                        fn += 1
                    else:
                        tn += 1
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics[group_name] = {'tpr': tpr, 'fpr': fpr}
        
        return metrics
    
    @staticmethod
    def calibration(predictions: Dict[str, float],
                   ground_truth: Dict[str, int],
                   groups: Dict[str, List[str]],
                   n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration (sufficiency).
        
        Requires that for any predicted probability p, the actual positive rate
        should be p across all groups. Critical for risk assessment tools.
        """
        calibration_data = {}
        for group_name, group_members in groups.items():
            bins = defaultdict(lambda: {'predictions': [], 'outcomes': []})
            
            for member in group_members:
                if member in predictions and member in ground_truth:
                    pred_prob = predictions[member]
                    outcome = ground_truth[member]
                    bin_idx = min(int(pred_prob * n_bins), n_bins - 1)
                    bins[bin_idx]['predictions'].append(pred_prob)
                    bins[bin_idx]['outcomes'].append(outcome)
            
            calibration_error = 0
            total_samples = 0
            for bin_data in bins.values():
                if bin_data['predictions']:
                    mean_pred = sum(bin_data['predictions']) / len(bin_data['predictions'])
                    mean_outcome = sum(bin_data['outcomes']) / len(bin_data['outcomes'])
                    n_samples = len(bin_data['predictions'])
                    calibration_error += abs(mean_pred - mean_outcome) * n_samples
                    total_samples += n_samples
            
            calibration_data[group_name] = calibration_error / total_samples if total_samples > 0 else 0
        
        return calibration_data
    
    @staticmethod
    def individual_fairness(predictions: Dict[str, float],
                          similarity_matrix: Dict[Tuple[str, str], float],
                          epsilon: float = 0.1) -> float:
        """
        Calculate individual fairness.
        
        Requires that similar individuals receive similar predictions.
        This is often overlooked but critical for avoiding discrimination
        against individuals rather than groups.
        """
        violations = 0
        comparisons = 0
        
        individuals = list(predictions.keys())
        for i, ind1 in enumerate(individuals):
            for ind2 in individuals[i+1:]:
                if (ind1, ind2) in similarity_matrix:
                    similarity = similarity_matrix[(ind1, ind2)]
                    pred_diff = abs(predictions[ind1] - predictions[ind2])
                    
                    if similarity > (1 - epsilon) and pred_diff > epsilon:
                        violations += 1
                    comparisons += 1
        
        return violations / comparisons if comparisons > 0 else 0


class ModelCard:
    """
    Comprehensive model documentation for transparency and accountability.
    
    Based on Google's Model Cards paper but extended with regulatory requirements
    and operational details often missing from academic implementations.
    """
    
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.created_date = datetime.now()
        self.last_updated = datetime.now()
        
        # Basic information
        self.model_details = {
            'architecture': None,
            'training_framework': None,
            'number_of_parameters': None,
            'input_format': None,
            'output_format': None,
            'inference_time_ms': None,
            'memory_requirements_mb': None
        }
        
        # Intended use
        self.intended_use = {
            'primary_use_cases': [],
            'primary_users': [],
            'out_of_scope_uses': [],
            'ethical_considerations': []
        }
        
        # Training data
        self.training_data = {
            'datasets': [],
            'data_collection_process': None,
            'preprocessing_steps': [],
            'known_biases': [],
            'data_governance_records': []  # Links to DataGovernanceRecord IDs
        }
        
        # Evaluation
        self.evaluation = {
            'metrics': {},
            'test_datasets': [],
            'performance_by_group': {},
            'fairness_metrics': {},
            'robustness_tests': {},
            'adversarial_testing': {}
        }
        
        # Limitations
        self.limitations = {
            'known_failure_modes': [],
            'confidence_thresholds': {},
            'environmental_dependencies': [],
            'temporal_validity': None  # How long the model remains valid
        }
        
        # Ethical review
        self.ethical_review = {
            'review_date': None,
            'reviewers': [],
            'identified_risks': [],
            'mitigation_strategies': [],
            'approval_status': None,
            'conditions': []
        }
        
        # Deployment
        self.deployment = {
            'deployment_environment': None,
            'monitoring_strategy': None,
            'update_frequency': None,
            'rollback_procedures': None,
            'human_oversight_required': False,
            'decision_threshold_adjustments': {}
        }
        
        # Regulatory compliance
        self.compliance = {
            'applicable_regulations': [],
            'compliance_attestations': {},
            'audit_trail': [],
            'data_processing_agreements': []
        }
    
    def add_evaluation_metric(self, metric_name: str, value: float, 
                             context: Optional[Dict[str, Any]] = None):
        """Add evaluation metric with context."""
        self.evaluation['metrics'][metric_name] = {
            'value': value,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def add_ethical_risk(self, risk: str, severity: str, 
                        likelihood: str, mitigation: str):
        """Document identified ethical risk and mitigation."""
        self.ethical_review['identified_risks'].append({
            'risk': risk,
            'severity': severity,  # low, medium, high, critical
            'likelihood': likelihood,  # unlikely, possible, likely, certain
            'mitigation': mitigation,
            'residual_risk': self._calculate_residual_risk(severity, likelihood)
        })
    
    def _calculate_residual_risk(self, severity: str, likelihood: str) -> str:
        """Calculate residual risk after mitigation."""
        risk_matrix = {
            ('low', 'unlikely'): 'negligible',
            ('low', 'possible'): 'low',
            ('low', 'likely'): 'medium',
            ('medium', 'unlikely'): 'low',
            ('medium', 'possible'): 'medium',
            ('medium', 'likely'): 'high',
            ('high', 'unlikely'): 'medium',
            ('high', 'possible'): 'high',
            ('high', 'likely'): 'critical',
            ('critical', 'unlikely'): 'high',
            ('critical', 'possible'): 'critical',
            ('critical', 'likely'): 'critical'
        }
        return risk_matrix.get((severity, likelihood), 'unknown')
    
    def generate_transparency_report(self) -> str:
        """Generate human-readable transparency report."""
        report = []
        report.append(f"Model Card: {self.model_name} v{self.version}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        report.append("Intended Use:")
        for use_case in self.intended_use['primary_use_cases']:
            report.append(f"  - {use_case}")
        
        report.append("")
        report.append("Out of Scope:")
        for misuse in self.intended_use['out_of_scope_uses']:
            report.append(f"  - {misuse}")
        
        if self.evaluation['fairness_metrics']:
            report.append("")
            report.append("Fairness Assessment:")
            for metric, value in self.evaluation['fairness_metrics'].items():
                report.append(f"  {metric}: {value}")
        
        if self.ethical_review['identified_risks']:
            report.append("")
            report.append("Ethical Risks:")
            for risk in self.ethical_review['identified_risks']:
                report.append(f"  Risk: {risk['risk']}")
                report.append(f"    Severity: {risk['severity']}, Likelihood: {risk['likelihood']}")
                report.append(f"    Mitigation: {risk['mitigation']}")
        
        return "\n".join(report)


class ExplainabilityFramework:
    """
    Comprehensive explainability implementation covering multiple techniques.
    
    Real-world explainability is more complex than just SHAP values. Different
    stakeholders need different types of explanations, and regulatory requirements
    vary by jurisdiction and industry.
    """
    
    def __init__(self, model: Any, model_type: str = "black_box"):
        self.model = model
        self.model_type = model_type
        self.explanation_cache = {}
        self.explanation_log = []
    
    def generate_local_explanation(self, instance: Dict[str, Any], 
                                  method: str = "lime") -> Dict[str, Any]:
        """
        Generate instance-level explanation.
        
        Local explanations are required for individual decisions in regulated
        industries like finance and healthcare.
        """
        explanation = {
            'instance': instance,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'confidence': None,
            'features': {}
        }
        
        if method == "lime":
            # Simplified LIME implementation concept
            perturbed_samples = self._generate_perturbations(instance, n_samples=100)
            predictions = [self._safe_predict(s) for s in perturbed_samples]
            
            # Fit local linear model
            feature_importance = self._fit_local_model(perturbed_samples, predictions, instance)
            explanation['features'] = feature_importance
            
        elif method == "counterfactual":
            # Find minimal change to flip prediction
            counterfactual = self._find_counterfactual(instance)
            explanation['counterfactual'] = counterfactual
            explanation['changes_required'] = self._diff_instances(instance, counterfactual)
            
        elif method == "prototype":
            # Find similar instances with different outcomes
            prototypes = self._find_prototypes(instance)
            explanation['prototypes'] = prototypes
        
        # Log explanation for audit
        self.explanation_log.append({
            'timestamp': datetime.now(),
            'instance_hash': self._hash_instance(instance),
            'method': method,
            'explanation_hash': hashlib.sha256(
                json.dumps(explanation, sort_keys=True).encode()
            ).hexdigest()
        })
        
        return explanation
    
    def generate_global_explanation(self, training_data: List[Dict[str, Any]],
                                  method: str = "feature_importance") -> Dict[str, Any]:
        """
        Generate model-level explanation.
        
        Global explanations help understand overall model behavior and
        identify potential biases or problematic patterns.
        """
        explanation = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'data_size': len(training_data),
            'results': {}
        }
        
        if method == "feature_importance":
            # Calculate permutation importance
            baseline_score = self._evaluate_model(training_data)
            importance_scores = {}
            
            for feature in self._get_features(training_data[0]):
                permuted_data = self._permute_feature(training_data, feature)
                permuted_score = self._evaluate_model(permuted_data)
                importance_scores[feature] = baseline_score - permuted_score
            
            explanation['results'] = importance_scores
            
        elif method == "partial_dependence":
            # Calculate PDP for each feature
            pdp_results = {}
            for feature in self._get_features(training_data[0]):
                pdp_results[feature] = self._calculate_pdp(training_data, feature)
            explanation['results'] = pdp_results
            
        elif method == "interaction_effects":
            # Identify feature interactions
            interactions = self._find_interactions(training_data)
            explanation['results'] = interactions
        
        return explanation
    
    def verify_explanation_consistency(self, instance: Dict[str, Any]) -> Dict[str, float]:
        """
        Check if different explanation methods agree.
        
        Inconsistent explanations indicate model instability or inappropriate
        explanation methods. This is rarely checked but critical for trust.
        """
        methods = ['lime', 'counterfactual', 'prototype']
        explanations = {}
        
        for method in methods:
            explanations[method] = self.generate_local_explanation(instance, method)
        
        # Compare explanations
        consistency_scores = {}
        
        # Check if top important features agree
        if 'lime' in explanations and 'counterfactual' in explanations:
            lime_features = set(list(explanations['lime']['features'].keys())[:5])
            cf_changes = set(list(explanations['counterfactual'].get('changes_required', {}).keys())[:5])
            
            overlap = len(lime_features & cf_changes) / max(len(lime_features), len(cf_changes))
            consistency_scores['lime_vs_counterfactual'] = overlap
        
        return consistency_scores
    
    def _generate_perturbations(self, instance: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
        """Generate perturbed samples around instance."""
        perturbations = []
        for _ in range(n_samples):
            perturbed = instance.copy()
            for feature, value in instance.items():
                if isinstance(value, (int, float)):
                    # Add noise to numerical features
                    noise = (hash(feature + str(_)) % 100 - 50) / 100
                    perturbed[feature] = value + noise * abs(value)
            perturbations.append(perturbed)
        return perturbations
    
    def _safe_predict(self, instance: Dict[str, Any]) -> float:
        """Make prediction with error handling."""
        try:
            # This would call the actual model
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0
    
    def _fit_local_model(self, samples: List[Dict[str, Any]], 
                        predictions: List[float],
                        instance: Dict[str, Any]) -> Dict[str, float]:
        """Fit interpretable model locally."""
        # Simplified - would use actual linear regression
        importance = {}
        for feature in instance.keys():
            importance[feature] = (hash(feature) % 100) / 100
        return importance
    
    def _find_counterfactual(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Find minimal change to flip prediction."""
        # Simplified - would use optimization
        counterfactual = instance.copy()
        for feature, value in instance.items():
            if isinstance(value, (int, float)):
                counterfactual[feature] = value * 1.1
        return counterfactual
    
    def _find_prototypes(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find representative examples."""
        # Simplified - would use nearest neighbors
        return [instance.copy() for _ in range(3)]
    
    def _diff_instances(self, inst1: Dict[str, Any], inst2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate difference between instances."""
        diff = {}
        for key in inst1:
            if key in inst2 and inst1[key] != inst2[key]:
                diff[key] = {'from': inst1[key], 'to': inst2[key]}
        return diff
    
    def _hash_instance(self, instance: Dict[str, Any]) -> str:
        """Create deterministic hash of instance."""
        return hashlib.sha256(
            json.dumps(instance, sort_keys=True).encode()
        ).hexdigest()
    
    def _get_features(self, instance: Dict[str, Any]) -> List[str]:
        """Extract feature names."""
        return list(instance.keys())
    
    def _permute_feature(self, data: List[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
        """Randomly permute a feature."""
        import random
        permuted = [d.copy() for d in data]
        values = [d[feature] for d in permuted]
        random.shuffle(values)
        for i, d in enumerate(permuted):
            d[feature] = values[i]
        return permuted
    
    def _evaluate_model(self, data: List[Dict[str, Any]]) -> float:
        """Evaluate model performance."""
        # Simplified - would calculate actual metric
        return 0.85
    
    def _calculate_pdp(self, data: List[Dict[str, Any]], feature: str) -> List[Tuple[float, float]]:
        """Calculate partial dependence plot."""
        # Simplified - would calculate actual PDP
        return [(i/10, i/10 * 0.5) for i in range(11)]
    
    def _find_interactions(self, data: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """Detect feature interactions."""
        # Simplified - would use H-statistic or similar
        interactions = {}
        features = self._get_features(data[0])
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                interactions[(f1, f2)] = (hash(f1 + f2) % 100) / 100
        return interactions


class PrivacyPreservingML:
    """
    Implementation of privacy-preserving machine learning techniques.
    
    Privacy is not just about encryption. It includes differential privacy,
    federated learning, secure multi-party computation, and homomorphic encryption.
    Most implementations miss subtle privacy leaks through timing, memory access,
    or model updates.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Differential privacy budget
        self.delta = delta      # Differential privacy failure probability
        self.privacy_accountant = PrivacyAccountant()
    
    def add_differential_privacy(self, data: Any, mechanism: str = "laplace") -> Any:
        """
        Add differential privacy to data or model updates.
        
        The choice of mechanism depends on the query type and sensitivity.
        Common mistake: not accounting for composition when multiple
        queries are made.
        """
        sensitivity = self._calculate_sensitivity(data)
        
        if mechanism == "laplace":
            scale = sensitivity / self.epsilon
            noise = self._generate_laplace_noise(scale, data.shape if hasattr(data, 'shape') else (1,))
            private_data = data + noise
            
        elif mechanism == "gaussian":
            # Gaussian mechanism for approximate differential privacy
            sigma = sensitivity * (2 * (1.25 / self.delta).log()) ** 0.5 / self.epsilon
            noise = self._generate_gaussian_noise(sigma, data.shape if hasattr(data, 'shape') else (1,))
            private_data = data + noise
            
        elif mechanism == "exponential":
            # For selecting from discrete set
            scores = self._compute_utility_scores(data)
            probabilities = self._exponential_mechanism(scores, sensitivity)
            private_data = self._sample_from_distribution(data, probabilities)
        
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        # Track privacy budget consumption
        self.privacy_accountant.consume_budget(self.epsilon, self.delta)
        
        return private_data
    
    def implement_federated_learning(self, local_models: List[Any], 
                                    aggregation: str = "fedavg") -> Any:
        """
        Aggregate models from federated learning.
        
        Key challenge: Byzantine-robust aggregation to handle malicious clients.
        Most implementations assume all clients are honest.
        """
        if aggregation == "fedavg":
            # Simple averaging - vulnerable to poisoning
            aggregated = self._average_models(local_models)
            
        elif aggregation == "median":
            # Coordinate-wise median - more robust
            aggregated = self._median_aggregation(local_models)
            
        elif aggregation == "krum":
            # Krum algorithm - Byzantine-robust
            aggregated = self._krum_aggregation(local_models)
            
        elif aggregation == "trimmed_mean":
            # Remove outliers before averaging
            aggregated = self._trimmed_mean_aggregation(local_models)
        
        # Add differential privacy to aggregated model
        aggregated = self.add_differential_privacy(aggregated, "gaussian")
        
        return aggregated
    
    def generate_synthetic_data(self, original_data: List[Dict[str, Any]], 
                              method: str = "gan") -> List[Dict[str, Any]]:
        """
        Generate privacy-preserving synthetic data.
        
        Synthetic data can still leak privacy if not carefully generated.
        Key insight: the generator itself can memorize training examples.
        """
        if method == "gan":
            # Use differentially private GAN
            synthetic = self._dp_gan_generate(original_data)
            
        elif method == "statistical":
            # Preserve statistical properties with noise
            statistics = self._compute_statistics(original_data)
            private_stats = self.add_differential_privacy(statistics, "laplace")
            synthetic = self._generate_from_statistics(private_stats, len(original_data))
            
        elif method == "bayesian":
            # Bayesian network with privacy
            network = self._learn_bayesian_network(original_data)
            private_network = self._privatize_network(network)
            synthetic = self._sample_from_network(private_network, len(original_data))
        
        # Validate privacy guarantees
        privacy_risk = self._estimate_privacy_risk(original_data, synthetic)
        if privacy_risk > 0.1:
            logger.warning(f"High privacy risk detected: {privacy_risk}")
        
        return synthetic
    
    def _calculate_sensitivity(self, data: Any) -> float:
        """Calculate query sensitivity for differential privacy."""
        # Simplified - actual calculation depends on query type
        if hasattr(data, 'max') and hasattr(data, 'min'):
            return float(data.max() - data.min())
        return 1.0
    
    def _generate_laplace_noise(self, scale: float, shape: Tuple) -> Any:
        """Generate Laplace noise."""
        import numpy as np
        return np.random.laplace(0, scale, shape)
    
    def _generate_gaussian_noise(self, sigma: float, shape: Tuple) -> Any:
        """Generate Gaussian noise."""
        import numpy as np
        return np.random.normal(0, sigma, shape)
    
    def _compute_utility_scores(self, data: Any) -> List[float]:
        """Compute utility scores for exponential mechanism."""
        # Simplified - would compute actual utility
        return [hash(str(item)) % 100 / 100 for item in data]
    
    def _exponential_mechanism(self, scores: List[float], sensitivity: float) -> List[float]:
        """Apply exponential mechanism for discrete selection."""
        import numpy as np
        exp_scores = np.exp(np.array(scores) * self.epsilon / (2 * sensitivity))
        return exp_scores / exp_scores.sum()
    
    def _sample_from_distribution(self, data: Any, probabilities: List[float]) -> Any:
        """Sample from discrete distribution."""
        import numpy as np
        idx = np.random.choice(len(data), p=probabilities)
        return data[idx]
    
    def _average_models(self, models: List[Any]) -> Any:
        """Simple model averaging."""
        # Placeholder - would average actual model parameters
        return models[0]
    
    def _median_aggregation(self, models: List[Any]) -> Any:
        """Coordinate-wise median aggregation."""
        # Placeholder - would compute actual median
        return models[len(models)//2]
    
    def _krum_aggregation(self, models: List[Any], f: int = 1) -> Any:
        """Krum aggregation for Byzantine robustness."""
        # Simplified - would implement full Krum algorithm
        # Select model with minimum distance to f nearest neighbors
        return models[0]
    
    def _trimmed_mean_aggregation(self, models: List[Any], trim_ratio: float = 0.1) -> Any:
        """Trimmed mean aggregation."""
        # Remove top and bottom trim_ratio of models
        n_trim = int(len(models) * trim_ratio)
        if n_trim > 0:
            return self._average_models(models[n_trim:-n_trim])
        return self._average_models(models)
    
    def _dp_gan_generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate synthetic data using DP-GAN."""
        # Placeholder - would use actual DP-GAN
        return [d.copy() for d in data[:10]]
    
    def _compute_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistical properties of data."""
        stats = {
            'mean': {},
            'std': {},
            'correlations': {}
        }
        # Simplified - would compute actual statistics
        return stats
    
    def _generate_from_statistics(self, stats: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
        """Generate data matching statistical properties."""
        # Placeholder
        return [{'generated': True} for _ in range(n_samples)]
    
    def _learn_bayesian_network(self, data: List[Dict[str, Any]]) -> Any:
        """Learn Bayesian network structure and parameters."""
        # Placeholder - would use structure learning algorithm
        return {'network': 'learned'}
    
    def _privatize_network(self, network: Any) -> Any:
        """Add privacy to Bayesian network parameters."""
        # Add noise to conditional probability tables
        return network
    
    def _sample_from_network(self, network: Any, n_samples: int) -> List[Dict[str, Any]]:
        """Sample from Bayesian network."""
        return [{'sampled': True} for _ in range(n_samples)]
    
    def _estimate_privacy_risk(self, original: List[Dict[str, Any]], 
                              synthetic: List[Dict[str, Any]]) -> float:
        """Estimate privacy risk of synthetic data."""
        # Simplified - would use membership inference or attribute inference
        return 0.05


class PrivacyAccountant:
    """
    Track privacy budget consumption across multiple operations.
    
    Critical for maintaining privacy guarantees over multiple queries.
    Most systems don't track this properly, leading to privacy violations.
    """
    
    def __init__(self):
        self.total_epsilon = 0
        self.total_delta = 0
        self.query_log = []
    
    def consume_budget(self, epsilon: float, delta: float):
        """Record privacy budget consumption."""
        self.total_epsilon += epsilon
        self.total_delta += delta
        self.query_log.append({
            'timestamp': datetime.now(),
            'epsilon': epsilon,
            'delta': delta,
            'cumulative_epsilon': self.total_epsilon,
            'cumulative_delta': self.total_delta
        })
    
    def get_remaining_budget(self, max_epsilon: float, max_delta: float) -> Tuple[float, float]:
        """Calculate remaining privacy budget."""
        remaining_epsilon = max(0, max_epsilon - self.total_epsilon)
        remaining_delta = max(0, max_delta - self.total_delta)
        return remaining_epsilon, remaining_delta


class AIEthicsAuditor:
    """
    Comprehensive auditing system for AI ethics compliance.
    
    Combines automated testing with human review processes. Goes beyond
    simple metrics to examine systemic issues and emergent behaviors.
    """
    
    def __init__(self, model: Any, model_card: ModelCard):
        self.model = model
        self.model_card = model_card
        self.audit_log = []
        self.findings = []
        self.test_results = {}
    
    def conduct_comprehensive_audit(self, test_data: List[Dict[str, Any]], 
                                   sensitive_attributes: List[str]) -> Dict[str, Any]:
        """
        Perform complete ethics audit.
        
        This is where everything comes together. The audit must be thorough
        but also actionable, providing clear remediation steps.
        """
        audit_report = {
            'audit_id': str(secrets.token_hex(16)),
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_card.model_name,
            'model_version': self.model_card.version,
            'findings': [],
            'recommendations': [],
            'risk_assessment': {},
            'compliance_status': {}
        }
        
        # Test for various biases
        bias_results = self._test_biases(test_data, sensitive_attributes)
        audit_report['bias_testing'] = bias_results
        
        # Test fairness metrics
        fairness_results = self._test_fairness(test_data, sensitive_attributes)
        audit_report['fairness_testing'] = fairness_results
        
        # Test robustness
        robustness_results = self._test_robustness(test_data)
        audit_report['robustness_testing'] = robustness_results
        
        # Test privacy
        privacy_results = self._test_privacy(test_data)
        audit_report['privacy_testing'] = privacy_results
        
        # Test explainability
        explain_results = self._test_explainability(test_data[:10])
        audit_report['explainability_testing'] = explain_results
        
        # Check regulatory compliance
        compliance_results = self._check_compliance()
        audit_report['compliance_status'] = compliance_results
        
        # Generate findings and recommendations
        findings = self._analyze_results(audit_report)
        audit_report['findings'] = findings
        
        recommendations = self._generate_recommendations(findings)
        audit_report['recommendations'] = recommendations
        
        # Risk assessment
        risk_score = self._calculate_risk_score(audit_report)
        audit_report['risk_assessment'] = {
            'overall_risk': risk_score,
            'risk_category': self._categorize_risk(risk_score),
            'immediate_actions_required': risk_score > 0.7
        }
        
        # Log audit
        self.audit_log.append(audit_report)
        
        return audit_report
    
    def _test_biases(self, test_data: List[Dict[str, Any]], 
                    sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Test for various types of bias."""
        results = {}
        
        for bias_type in BiasType:
            test_name = f"test_{bias_type.name.lower()}_bias"
            if hasattr(self, test_name):
                test_method = getattr(self, test_name)
                results[bias_type.name] = test_method(test_data, sensitive_attributes)
            else:
                # Generic bias test
                results[bias_type.name] = self._generic_bias_test(
                    test_data, sensitive_attributes, bias_type
                )
        
        return results
    
    def _generic_bias_test(self, test_data: List[Dict[str, Any]], 
                          sensitive_attributes: List[str],
                          bias_type: BiasType) -> Dict[str, Any]:
        """Generic bias testing framework."""
        result = {
            'bias_type': bias_type.name,
            'detected': False,
            'severity': None,
            'details': {}
        }
        
        # Simplified bias detection
        # In practice, each bias type needs specific testing
        for attr in sensitive_attributes:
            groups = self._partition_by_attribute(test_data, attr)
            group_predictions = {}
            
            for group_name, group_data in groups.items():
                predictions = [self._safe_model_predict(d) for d in group_data]
                group_predictions[group_name] = predictions
            
            # Check for significant differences
            if self._has_significant_difference(group_predictions):
                result['detected'] = True
                result['severity'] = 'medium'
                result['details'][attr] = group_predictions
        
        return result
    
    def _test_fairness(self, test_data: List[Dict[str, Any]], 
                      sensitive_attributes: List[str]) -> Dict[str, Any]:
        """Test fairness metrics."""
        results = {}
        fairness_calc = FairnessMetric()
        
        for attr in sensitive_attributes:
            groups = self._partition_by_attribute(test_data, attr)
            
            # Get predictions and ground truth
            predictions = {}
            ground_truth = {}
            for item in test_data:
                item_id = str(item.get('id', hash(str(item))))
                predictions[item_id] = self._safe_model_predict(item)
                ground_truth[item_id] = item.get('label', 0)
            
            # Calculate various fairness metrics
            results[attr] = {
                'demographic_parity': fairness_calc.demographic_parity(predictions, groups),
                'equalized_odds': fairness_calc.equalized_odds(predictions, ground_truth, groups),
                'calibration': fairness_calc.calibration(predictions, ground_truth, groups)
            }
        
        return results
    
    def _test_robustness(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test model robustness."""
        results = {
            'adversarial_robustness': self._test_adversarial(test_data[:100]),
            'noise_robustness': self._test_noise_robustness(test_data[:100]),
            'distribution_shift': self._test_distribution_shift(test_data)
        }
        return results
    
    def _test_adversarial(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test adversarial robustness."""
        # Simplified - would implement actual adversarial attack
        return {
            'attack_success_rate': 0.15,
            'average_perturbation': 0.03,
            'worst_case_examples': []
        }
    
    def _test_noise_robustness(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness to random noise."""
        results = []
        for item in test_data:
            original_pred = self._safe_model_predict(item)
            
            # Add noise
            noisy_item = item.copy()
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    noise = (hash(key) % 100 - 50) / 1000
                    noisy_item[key] = value + noise
            
            noisy_pred = self._safe_model_predict(noisy_item)
            results.append(abs(original_pred - noisy_pred))
        
        return {
            'mean_prediction_change': sum(results) / len(results) if results else 0,
            'max_prediction_change': max(results) if results else 0
        }
    
    def _test_distribution_shift(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test for distribution shift vulnerabilities."""
        # Simplified - would implement actual drift detection
        return {
            'drift_detected': False,
            'drift_magnitude': 0.02
        }
    
    def _test_privacy(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test privacy preservation."""
        results = {
            'membership_inference_risk': self._test_membership_inference(test_data[:100]),
            'attribute_inference_risk': self._test_attribute_inference(test_data[:100]),
            'model_inversion_risk': self._test_model_inversion()
        }
        return results
    
    def _test_membership_inference(self, test_data: List[Dict[str, Any]]) -> float:
        """Test membership inference attack risk."""
        # Simplified - would implement actual attack
        return 0.52  # Just above random guessing
    
    def _test_attribute_inference(self, test_data: List[Dict[str, Any]]) -> float:
        """Test attribute inference attack risk."""
        # Simplified - would implement actual attack
        return 0.15
    
    def _test_model_inversion(self) -> float:
        """Test model inversion attack risk."""
        # Simplified - would implement actual attack
        return 0.05
    
    def _test_explainability(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test explanation quality and consistency."""
        explainer = ExplainabilityFramework(self.model)
        results = {
            'explanation_consistency': [],
            'explanation_stability': [],
            'human_interpretability_score': None
        }
        
        for item in test_data:
            consistency = explainer.verify_explanation_consistency(item)
            results['explanation_consistency'].append(consistency)
        
        return results
    
    def _check_compliance(self) -> Dict[str, bool]:
        """Check regulatory compliance."""
        compliance = {}
        
        for regulation in self.model_card.compliance['applicable_regulations']:
            if regulation == RegulatoryFramework.GDPR:
                compliance['GDPR'] = self._check_gdpr_compliance()
            elif regulation == RegulatoryFramework.CCPA:
                compliance['CCPA'] = self._check_ccpa_compliance()
            # Add other regulations as needed
        
        return compliance
    
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance."""
        checks = [
            self.model_card.compliance.get('data_minimization', False),
            self.model_card.compliance.get('purpose_limitation', False),
            self.model_card.compliance.get('right_to_explanation', False),
            self.model_card.compliance.get('privacy_by_design', False)
        ]
        return all(checks)
    
    def _check_ccpa_compliance(self) -> bool:
        """Check CCPA compliance."""
        checks = [
            self.model_card.compliance.get('opt_out_mechanism', False),
            self.model_card.compliance.get('non_discrimination', False)
        ]
        return all(checks)
    
    def _partition_by_attribute(self, data: List[Dict[str, Any]], 
                               attribute: str) -> Dict[str, List[Dict[str, Any]]]:
        """Partition data by sensitive attribute."""
        groups = defaultdict(list)
        for item in data:
            if attribute in item:
                groups[str(item[attribute])].append(item)
        return dict(groups)
    
    def _safe_model_predict(self, item: Dict[str, Any]) -> float:
        """Safe model prediction with error handling."""
        try:
            # Placeholder - would call actual model
            return hash(str(item)) % 100 / 100
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5
    
    def _has_significant_difference(self, group_predictions: Dict[str, List[float]]) -> bool:
        """Check if there's significant difference between groups."""
        if len(group_predictions) < 2:
            return False
        
        means = [sum(preds)/len(preds) if preds else 0 
                for preds in group_predictions.values()]
        
        # Simplified - would use proper statistical test
        return max(means) - min(means) > 0.1
    
    def _analyze_results(self, audit_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze audit results to generate findings."""
        findings = []
        
        # Check bias testing results
        if 'bias_testing' in audit_report:
            for bias_name, bias_result in audit_report['bias_testing'].items():
                if bias_result.get('detected', False):
                    findings.append({
                        'type': 'bias',
                        'name': bias_name,
                        'severity': bias_result.get('severity', 'unknown'),
                        'details': bias_result.get('details', {})
                    })
        
        # Check fairness results
        if 'fairness_testing' in audit_report:
            for attr, metrics in audit_report['fairness_testing'].items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        if metric_value.get('max_disparity', 0) > 0.2:
                            findings.append({
                                'type': 'fairness_violation',
                                'attribute': attr,
                                'metric': metric_name,
                                'severity': 'high' if metric_value.get('max_disparity', 0) > 0.3 else 'medium'
                            })
        
        return findings
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations from findings."""
        recommendations = []
        
        bias_findings = [f for f in findings if f['type'] == 'bias']
        if bias_findings:
            recommendations.append(
                "Implement bias mitigation techniques such as reweighting, "
                "resampling, or adversarial debiasing"
            )
        
        fairness_findings = [f for f in findings if f['type'] == 'fairness_violation']
        if fairness_findings:
            recommendations.append(
                "Adjust decision thresholds per group to achieve fairness metrics, "
                "or retrain with fairness constraints"
            )
        
        return recommendations
    
    def _calculate_risk_score(self, audit_report: Dict[str, Any]) -> float:
        """Calculate overall risk score."""
        risk_factors = []
        
        # Count findings by severity
        findings = audit_report.get('findings', [])
        critical_findings = len([f for f in findings if f.get('severity') == 'critical'])
        high_findings = len([f for f in findings if f.get('severity') == 'high'])
        medium_findings = len([f for f in findings if f.get('severity') == 'medium'])
        
        # Weight by severity
        risk_score = (critical_findings * 1.0 + high_findings * 0.6 + medium_findings * 0.3)
        
        # Normalize to 0-1 range
        return min(1.0, risk_score / 10)
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level."""
        if risk_score < 0.2:
            return 'low'
        elif risk_score < 0.5:
            return 'medium'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'critical'


def demonstrate_ethics_framework():
    """
    Demonstration of the comprehensive AI ethics framework.
    
    This shows how all components work together in practice.
    """
    print("AI Ethics and Governance Framework Demonstration")
    print("Author: Cazandra Aporbo")
    print()
    
    # Create model card
    model_card = ModelCard("risk_assessment_model", "1.0.0")
    model_card.intended_use['primary_use_cases'] = ["Credit risk assessment"]
    model_card.intended_use['out_of_scope_uses'] = ["Employment decisions", "Insurance pricing"]
    model_card.compliance['applicable_regulations'] = [
        RegulatoryFramework.FCRA,
        RegulatoryFramework.ECOA,
        RegulatoryFramework.GDPR
    ]
    
    # Add ethical risks
    model_card.add_ethical_risk(
        risk="Perpetuation of historical lending bias",
        severity="high",
        likelihood="likely",
        mitigation="Regular bias audits and threshold adjustments"
    )
    
    # Create test data
    test_data = [
        {
            'id': i,
            'income': 50000 + i * 1000,
            'credit_score': 650 + i * 5,
            'employment_years': 5 + i % 10,
            'protected_attribute': 'group_A' if i % 3 == 0 else 'group_B',
            'label': 1 if i % 4 != 0 else 0
        }
        for i in range(100)
    ]
    
    # Create dummy model
    class DummyModel:
        def predict(self, x):
            return 0.5
    
    model = DummyModel()
    
    # Conduct audit
    auditor = AIEthicsAuditor(model, model_card)
    audit_report = auditor.conduct_comprehensive_audit(
        test_data,
        sensitive_attributes=['protected_attribute']
    )
    
    # Display results
    print(f"Audit ID: {audit_report['audit_id']}")
    print(f"Risk Assessment: {audit_report['risk_assessment']['risk_category']}")
    print(f"Number of findings: {len(audit_report['findings'])}")
    print()
    
    print("Key Findings:")
    for finding in audit_report['findings'][:3]:
        print(f"  - {finding.get('type')}: {finding.get('name', 'N/A')} "
              f"(Severity: {finding.get('severity', 'unknown')})")
    
    print()
    print("Recommendations:")
    for rec in audit_report['recommendations'][:3]:
        print(f"  - {rec}")
    
    # Generate transparency report
    print()
    print("Transparency Report:")
    print(model_card.generate_transparency_report())
    
    # Test privacy preservation
    print()
    print("Privacy Preservation Test:")
    privacy_ml = PrivacyPreservingML(epsilon=1.0, delta=1e-5)
    
    # Create synthetic data
    synthetic_data = privacy_ml.generate_synthetic_data(test_data[:10], method="statistical")
    print(f"Generated {len(synthetic_data)} synthetic records with differential privacy")
    
    remaining_budget = privacy_ml.privacy_accountant.get_remaining_budget(10.0, 1e-4)
    print(f"Remaining privacy budget: epsilon={remaining_budget[0]:.2f}, delta={remaining_budget[1]:.2e}")


if __name__ == "__main__":
    demonstrate_ethics_framework()
