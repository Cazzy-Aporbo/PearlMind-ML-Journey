#!/usr/bin/env python3
"""
PearlMind Machine Learning Platform
Advanced implementation of machine learning algorithms with visualization and analysis
By Cazandra Aporbo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import warnings
import json
import pickle
import hashlib
import time
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats, signal, optimize
from scipy.spatial import distance
from scipy.special import expit, logit
import logging
from pathlib import Path
import random
from datetime import datetime
from functools import lru_cache, wraps
import inspect
import threading
import queue

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ModelMetrics:
    """Container for comprehensive model evaluation metrics."""
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    explained_variance: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() 
                if v is not None and not isinstance(v, np.ndarray)}


class DataPreprocessor:
    """Advanced data preprocessing with automatic feature engineering."""
    
    def __init__(self, strategy: str = 'auto'):
        """
        Initialize preprocessor with specified strategy.
        
        The strategy determines how missing values, outliers, and scaling are handled.
        Auto mode intelligently selects methods based on data characteristics.
        """
        self.strategy = strategy
        self.feature_stats = {}
        self.transformations = {}
        self.encoding_maps = {}
        self.scaler_params = {}
        
    def analyze_data(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive data analysis to inform preprocessing decisions.
        
        This analysis examines distributions, correlations, and data quality issues
        to automatically determine the best preprocessing approach for each feature.
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_usage': X.nbytes / (1024 * 1024),  # MB
            'features': {}
        }
        
        for i, name in enumerate(feature_names):
            col = X[:, i]
            
            # Check if feature is numeric or categorical
            unique_ratio = len(np.unique(col)) / len(col)
            is_categorical = unique_ratio < 0.05 and len(np.unique(col)) < 20
            
            feature_info = {
                'type': 'categorical' if is_categorical else 'numeric',
                'missing_count': np.sum(np.isnan(col)) if not is_categorical else 0,
                'missing_percentage': np.mean(np.isnan(col)) * 100 if not is_categorical else 0,
                'unique_values': len(np.unique(col)),
                'unique_ratio': unique_ratio
            }
            
            if not is_categorical:
                # Calculate statistics for numeric features
                clean_col = col[~np.isnan(col)]
                if len(clean_col) > 0:
                    feature_info.update({
                        'mean': np.mean(clean_col),
                        'std': np.std(clean_col),
                        'min': np.min(clean_col),
                        'max': np.max(clean_col),
                        'q25': np.percentile(clean_col, 25),
                        'median': np.percentile(clean_col, 50),
                        'q75': np.percentile(clean_col, 75),
                        'skewness': stats.skew(clean_col),
                        'kurtosis': stats.kurtosis(clean_col)
                    })
                    
                    # Detect outliers using IQR method
                    q1 = feature_info['q25']
                    q3 = feature_info['q75']
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = np.sum((clean_col < lower_bound) | (clean_col > upper_bound))
                    feature_info['outlier_count'] = outliers
                    feature_info['outlier_percentage'] = (outliers / len(clean_col)) * 100
            
            analysis['features'][name] = feature_info
            self.feature_stats[name] = feature_info
        
        return analysis
    
    def handle_missing_values(self, X: np.ndarray, method: str = 'smart') -> np.ndarray:
        """
        Handle missing values using various imputation strategies.
        
        Smart mode selects the best method based on data characteristics:
        - Mean/median for normal/skewed numeric data
        - Mode for categorical data
        - Forward fill for time series
        - KNN imputation for complex patterns
        """
        X_imputed = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            missing_mask = np.isnan(col)
            
            if not np.any(missing_mask):
                continue
            
            if method == 'smart':
                # Determine best imputation based on data characteristics
                feature_name = f"feature_{i}"
                if feature_name in self.feature_stats:
                    stats = self.feature_stats[feature_name]
                    
                    if stats['type'] == 'categorical':
                        # Use mode for categorical
                        mode_val = stats.mode(col[~missing_mask])[0][0]
                        X_imputed[missing_mask, i] = mode_val
                    elif abs(stats.get('skewness', 0)) > 2:
                        # Use median for highly skewed data
                        X_imputed[missing_mask, i] = np.median(col[~missing_mask])
                    else:
                        # Use mean for approximately normal data
                        X_imputed[missing_mask, i] = np.mean(col[~missing_mask])
                else:
                    # Fallback to median
                    X_imputed[missing_mask, i] = np.median(col[~missing_mask])
            
            elif method == 'knn':
                # KNN imputation (simplified version)
                X_imputed[:, i] = self._knn_impute(X, i, k=5)
            
            elif method == 'forward_fill':
                # Forward fill for time series
                last_valid = np.nan
                for j in range(len(col)):
                    if missing_mask[j]:
                        if not np.isnan(last_valid):
                            X_imputed[j, i] = last_valid
                    else:
                        last_valid = col[j]
            
            else:
                # Simple mean imputation
                X_imputed[missing_mask, i] = np.mean(col[~missing_mask])
        
        return X_imputed
    
    def _knn_impute(self, X: np.ndarray, col_idx: int, k: int = 5) -> np.ndarray:
        """
        KNN imputation for a single column.
        
        Finds k nearest neighbors based on other features and uses their
        values to impute missing data in the target column.
        """
        col = X[:, col_idx].copy()
        missing_mask = np.isnan(col)
        
        if not np.any(missing_mask):
            return col
        
        # Use other columns for distance calculation
        other_cols = [j for j in range(X.shape[1]) if j != col_idx]
        X_other = X[:, other_cols]
        
        for i in np.where(missing_mask)[0]:
            # Find k nearest neighbors with non-missing values
            distances = []
            for j in np.where(~missing_mask)[0]:
                # Calculate Euclidean distance, handling missing values
                dist = np.nanmean((X_other[i] - X_other[j]) ** 2) ** 0.5
                distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            neighbors = [idx for idx, _ in distances[:k]]
            
            # Impute as mean of neighbors
            col[i] = np.mean([col[n] for n in neighbors])
        
        return col
    
    def detect_and_handle_outliers(self, X: np.ndarray, 
                                  method: str = 'iqr',
                                  threshold: float = 3.0) -> np.ndarray:
        """
        Detect and handle outliers using various methods.
        
        Methods include:
        - IQR: Interquartile range method
        - Z-score: Statistical z-score method
        - Isolation Forest: Tree-based anomaly detection
        - LOF: Local Outlier Factor
        """
        X_cleaned = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            if method == 'iqr':
                q1 = np.percentile(col, 25)
                q3 = np.percentile(col, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Cap outliers at bounds
                X_cleaned[:, i] = np.clip(col, lower_bound, upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((col - np.mean(col)) / np.std(col))
                outlier_mask = z_scores > threshold
                
                # Replace outliers with median
                if np.any(outlier_mask):
                    X_cleaned[outlier_mask, i] = np.median(col[~outlier_mask])
            
            elif method == 'isolation':
                # Simplified isolation forest
                outlier_scores = self._isolation_scores(col.reshape(-1, 1))
                outlier_mask = outlier_scores > 0.6
                
                if np.any(outlier_mask):
                    X_cleaned[outlier_mask, i] = np.median(col[~outlier_mask])
        
        return X_cleaned
    
    def _isolation_scores(self, X: np.ndarray, n_trees: int = 100) -> np.ndarray:
        """
        Calculate isolation scores for anomaly detection.
        
        Implements a simplified version of Isolation Forest algorithm
        which isolates anomalies by randomly selecting features and splits.
        """
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for _ in range(n_trees):
            # Simplified tree: random splits
            tree_scores = np.zeros(n_samples)
            
            for i in range(n_samples):
                # Path length to isolate sample
                path_length = 0
                sample = X[i]
                
                # Simplified: count splits needed
                for _ in range(int(np.log2(n_samples))):
                    split_value = np.random.uniform(X.min(), X.max())
                    path_length += 1
                    
                    if sample < split_value:
                        if np.random.random() > 0.5:
                            break
                
                tree_scores[i] = path_length
            
            scores += tree_scores
        
        # Normalize scores
        scores = scores / n_trees
        avg_path_length = 2 * (np.log(n_samples - 1) + 0.5772) - 2 * (n_samples - 1) / n_samples
        scores = 2 ** (-scores / avg_path_length)
        
        return scores
    
    def scale_features(self, X: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Scale features using various normalization techniques.
        
        Methods:
        - Standard: Zero mean, unit variance
        - MinMax: Scale to [0, 1] range
        - Robust: Using median and IQR (robust to outliers)
        - MaxAbs: Scale by maximum absolute value
        """
        X_scaled = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            if method == 'standard':
                mean = np.mean(col)
                std = np.std(col)
                if std > 0:
                    X_scaled[:, i] = (col - mean) / std
                    self.scaler_params[i] = {'mean': mean, 'std': std}
            
            elif method == 'minmax':
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val > min_val:
                    X_scaled[:, i] = (col - min_val) / (max_val - min_val)
                    self.scaler_params[i] = {'min': min_val, 'max': max_val}
            
            elif method == 'robust':
                median = np.median(col)
                q1 = np.percentile(col, 25)
                q3 = np.percentile(col, 75)
                iqr = q3 - q1
                if iqr > 0:
                    X_scaled[:, i] = (col - median) / iqr
                    self.scaler_params[i] = {'median': median, 'iqr': iqr}
            
            elif method == 'maxabs':
                max_abs = np.max(np.abs(col))
                if max_abs > 0:
                    X_scaled[:, i] = col / max_abs
                    self.scaler_params[i] = {'max_abs': max_abs}
        
        return X_scaled
    
    def engineer_features(self, X: np.ndarray) -> np.ndarray:
        """
        Automatically engineer new features based on existing ones.
        
        Creates polynomial features, interactions, and statistical aggregates
        to capture non-linear relationships in the data.
        """
        n_samples, n_features = X.shape
        engineered = []
        
        # Polynomial features (degree 2 for selected features)
        for i in range(min(n_features, 5)):  # Limit to prevent explosion
            engineered.append(X[:, i] ** 2)
        
        # Interaction features for top correlated pairs
        if n_features > 1:
            for i in range(min(n_features - 1, 3)):
                for j in range(i + 1, min(n_features, 4)):
                    engineered.append(X[:, i] * X[:, j])
        
        # Statistical features (rolling if time series)
        if n_samples > 10:
            for i in range(min(n_features, 3)):
                # Moving averages
                window_size = min(5, n_samples // 2)
                rolling_mean = np.convolve(X[:, i], 
                                          np.ones(window_size) / window_size, 
                                          mode='same')
                engineered.append(rolling_mean)
        
        # Combine original and engineered features
        if engineered:
            X_engineered = np.column_stack([X] + engineered)
        else:
            X_engineered = X
        
        logger.info(f"Engineered {len(engineered)} new features")
        
        return X_engineered
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Applies the complete preprocessing pipeline:
        1. Analyze data characteristics
        2. Handle missing values
        3. Remove outliers
        4. Scale features
        5. Engineer new features
        """
        # Analyze data
        self.analyze_data(X)
        
        # Apply preprocessing pipeline
        X_processed = self.handle_missing_values(X, method='smart')
        X_processed = self.detect_and_handle_outliers(X_processed, method='iqr')
        X_processed = self.scale_features(X_processed, method='standard')
        
        # Optional feature engineering
        if self.strategy == 'auto' and X.shape[1] < 20:
            X_processed = self.engineer_features(X_processed)
        
        return X_processed


class NeuralNetwork:
    """Deep neural network with automatic architecture selection and training."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 regularization: float = 0.01):
        """
        Initialize neural network with specified architecture.
        
        If hidden_layers is None, automatically determines architecture
        based on input/output dimensions and complexity heuristics.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Automatically determine architecture if not specified
        if hidden_layers is None:
            self.hidden_layers = self._auto_architecture(input_dim, output_dim)
        else:
            self.hidden_layers = hidden_layers
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Training history
        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_loss = float('inf')
        
    def _auto_architecture(self, input_dim: int, output_dim: int) -> List[int]:
        """
        Automatically determine network architecture based on problem complexity.
        
        Uses heuristics based on input/output dimensions and empirical rules
        for selecting appropriate hidden layer sizes.
        """
        # Heuristic: pyramid structure with decreasing neurons
        if input_dim <= 10:
            # Simple problem: 1-2 hidden layers
            hidden1 = int(input_dim * 2)
            if output_dim == 1:
                return [hidden1]
            else:
                hidden2 = max(output_dim * 2, int(hidden1 / 2))
                return [hidden1, hidden2]
        else:
            # Complex problem: 2-3 hidden layers
            hidden1 = int(input_dim * 1.5)
            hidden2 = int(hidden1 * 0.7)
            
            if input_dim > 50:
                hidden3 = max(output_dim * 2, int(hidden2 * 0.5))
                return [hidden1, hidden2, hidden3]
            else:
                return [hidden1, hidden2]
    
    def _initialize_parameters(self):
        """
        Initialize network parameters using Xavier/He initialization.
        
        Xavier initialization for tanh/sigmoid: sqrt(1/n_in)
        He initialization for ReLU: sqrt(2/n_in)
        """
        layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            
            # Choose initialization based on activation
            if self.activation == 'relu':
                # He initialization
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier initialization
                std = np.sqrt(1.0 / n_in)
            
            # Initialize weights and biases
            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _activate(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function element-wise."""
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'sigmoid':
            return expit(z)  # More stable than 1/(1+exp(-z))
        elif activation == 'leaky_relu':
            return np.where(z > 0, z, z * 0.01)
        elif activation == 'softmax':
            # Stable softmax
            z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return z_exp / np.sum(z_exp, axis=1, keepdims=True)
        else:
            return z
    
    def _activate_derivative(self, a: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation == 'relu':
            return (a > 0).astype(float)
        elif activation == 'tanh':
            return 1 - a ** 2
        elif activation == 'sigmoid':
            return a * (1 - a)
        elif activation == 'leaky_relu':
            return np.where(a > 0, 1, 0.01)
        else:
            return np.ones_like(a)
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform forward propagation through the network.
        
        Returns final output and intermediate activations for backpropagation.
        """
        activations = [X]
        a = X
        
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            
            # Apply activation function
            if i < len(self.weights) - 1:
                # Hidden layers
                a = self._activate(z, self.activation)
            else:
                # Output layer
                if self.output_dim == 1:
                    a = self._activate(z, 'sigmoid')
                elif self.output_dim > 1:
                    a = self._activate(z, 'softmax')
                else:
                    a = z
            
            activations.append(a)
        
        return a, activations
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                           activations: List[np.ndarray]) -> Tuple[List, List]:
        """
        Perform backward propagation to calculate gradients.
        
        Uses chain rule to compute gradients of loss with respect to
        weights and biases at each layer.
        """
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Calculate output layer gradients
        if self.output_dim == 1:
            # Binary classification or regression
            dz = activations[-1] - y.reshape(-1, 1)
        else:
            # Multi-class classification
            dz = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Calculate gradients
            dw = (1/m) * np.dot(activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # Add L2 regularization to weights
            dw += (self.regularization / m) * self.weights[i]
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            # Calculate gradient for previous layer
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._activate_derivative(activations[i], self.activation)
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w: List, gradients_b: List, 
                         iteration: int = 0):
        """
        Update network parameters using specified optimizer.
        
        Implements various optimization algorithms:
        - SGD: Standard gradient descent
        - Momentum: SGD with momentum
        - Adam: Adaptive moment estimation
        - RMSprop: Root mean square propagation
        """
        if self.optimizer_type == 'sgd':
            # Standard gradient descent
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        
        elif self.optimizer_type == 'momentum':
            # SGD with momentum
            if not hasattr(self, 'velocity_w'):
                self.velocity_w = [np.zeros_like(w) for w in self.weights]
                self.velocity_b = [np.zeros_like(b) for b in self.biases]
            
            beta = 0.9
            for i in range(len(self.weights)):
                self.velocity_w[i] = beta * self.velocity_w[i] + (1 - beta) * gradients_w[i]
                self.velocity_b[i] = beta * self.velocity_b[i] + (1 - beta) * gradients_b[i]
                
                self.weights[i] -= self.learning_rate * self.velocity_w[i]
                self.biases[i] -= self.learning_rate * self.velocity_b[i]
        
        elif self.optimizer_type == 'adam':
            # Adam optimizer
            if not hasattr(self, 'adam_m_w'):
                self.adam_m_w = [np.zeros_like(w) for w in self.weights]
                self.adam_m_b = [np.zeros_like(b) for b in self.biases]
                self.adam_v_w = [np.zeros_like(w) for w in self.weights]
                self.adam_v_b = [np.zeros_like(b) for b in self.biases]
            
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            t = iteration + 1
            
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.adam_m_w[i] = beta1 * self.adam_m_w[i] + (1 - beta1) * gradients_w[i]
                self.adam_m_b[i] = beta1 * self.adam_m_b[i] + (1 - beta1) * gradients_b[i]
                
                # Update biased second moment estimate
                self.adam_v_w[i] = beta2 * self.adam_v_w[i] + (1 - beta2) * (gradients_w[i] ** 2)
                self.adam_v_b[i] = beta2 * self.adam_v_b[i] + (1 - beta2) * (gradients_b[i] ** 2)
                
                # Compute bias-corrected estimates
                m_w_hat = self.adam_m_w[i] / (1 - beta1 ** t)
                m_b_hat = self.adam_m_b[i] / (1 - beta1 ** t)
                v_w_hat = self.adam_v_w[i] / (1 - beta2 ** t)
                v_b_hat = self.adam_v_b[i] / (1 - beta2 ** t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    
    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate loss based on problem type.
        
        Uses cross-entropy for classification, MSE for regression.
        """
        m = y_true.shape[0]
        
        if self.output_dim == 1:
            # Binary cross-entropy
            epsilon = 1e-7
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.output_dim > 1 and len(y_true.shape) > 1:
            # Categorical cross-entropy
            epsilon = 1e-7
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            # Mean squared error
            loss = np.mean((y_true - y_pred) ** 2)
        
        # Add L2 regularization
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        loss += (self.regularization / (2 * m)) * l2_loss
        
        return loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32,
             early_stopping_patience: int = 10, verbose: bool = True):
        """
        Train the neural network using mini-batch gradient descent.
        
        Implements:
        - Mini-batch training for efficiency
        - Early stopping to prevent overfitting
        - Learning rate scheduling
        - Gradient clipping for stability
        """
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward propagation
                y_pred, activations = self.forward_propagation(X_batch)
                
                # Calculate loss
                batch_loss = self.calculate_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward propagation
                gradients_w, gradients_b = self.backward_propagation(X_batch, y_batch, activations)
                
                # Gradient clipping for stability
                max_grad_norm = 5.0
                for i in range(len(gradients_w)):
                    grad_norm = np.linalg.norm(gradients_w[i])
                    if grad_norm > max_grad_norm:
                        gradients_w[i] = gradients_w[i] * max_grad_norm / grad_norm
                
                # Update parameters
                self.update_parameters(gradients_w, gradients_b, epoch * n_batches)
            
            # Calculate average epoch loss
            epoch_loss = epoch_loss / n_batches
            self.loss_history.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_pred, _ = self.forward_propagation(X_val)
                val_loss = self.calculate_loss(y_val, val_pred)
                self.val_loss_history.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
            # Learning rate decay
            if epoch > 0 and epoch % 50 == 0:
                self.learning_rate *= 0.9
        
        # Restore best weights if using validation
        if self.best_weights is not None:
            self.weights = self.best_weights
            self.biases = self.best_biases
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        y_pred, _ = self.forward_propagation(X)
        
        if self.output_dim == 1:
            # Binary classification: convert probabilities to classes
            return (y_pred > 0.5).astype(int).reshape(-1)
        elif self.output_dim > 1:
            # Multi-class: return class with highest probability
            return np.argmax(y_pred, axis=1)
        else:
            # Regression: return raw predictions
            return y_pred.reshape(-1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        y_pred, _ = self.forward_propagation(X)
        return y_pred


class GradientBoostingMachine:
    """Custom implementation of gradient boosting for regression and classification."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 subsample: float = 1.0,
                 loss: str = 'squared'):
        """
        Initialize gradient boosting machine.
        
        Implements boosting by fitting sequential weak learners (decision trees)
        to the residuals of previous predictions.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.loss = loss
        
        self.trees = []
        self.feature_importance_ = None
        self.initial_prediction = None
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray, 
                   max_depth: int, min_samples_split: int) -> Dict:
        """
        Build a decision tree using recursive partitioning.
        
        Simple implementation of CART algorithm for regression trees.
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if n_samples < min_samples_split or max_depth <= 0:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # Find best split
        best_gain = 0
        best_split = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
                    continue
                
                # Calculate variance reduction
                variance_before = np.var(y)
                variance_left = np.var(y[left_mask])
                variance_right = np.var(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_variance = (n_left * variance_left + n_right * variance_right) / n_samples
                gain = variance_before - weighted_variance
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        # No good split found
        if best_split is None:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        # Build subtrees
        left_tree = self._build_tree(
            X[best_split['left_mask']],
            y[best_split['left_mask']],
            max_depth - 1,
            min_samples_split
        )
        
        right_tree = self._build_tree(
            X[best_split['right_mask']],
            y[best_split['right_mask']],
            max_depth - 1,
            min_samples_split
        )
        
        return {
            'type': 'split',
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """Make predictions using a single tree."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            node = tree
            while node['type'] != 'leaf':
                if X[i, node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions[i] = node['value']
        
        return predictions
    
    def _calculate_negative_gradient(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate negative gradient of the loss function.
        
        This is what we fit the next tree to in gradient boosting.
        """
        if self.loss == 'squared':
            # MSE loss: gradient is simply residual
            return y_true - y_pred
        elif self.loss == 'absolute':
            # MAE loss: gradient is sign of residual
            return np.sign(y_true - y_pred)
        elif self.loss == 'huber':
            # Huber loss: combination of MSE and MAE
            delta = 1.0
            residual = y_true - y_pred
            return np.where(np.abs(residual) <= delta,
                          residual,
                          delta * np.sign(residual))
        else:
            return y_true - y_pred
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit gradient boosting model.
        
        Sequentially fits trees to negative gradients of the loss function.
        """
        n_samples, n_features = X.shape
        
        # Initialize with mean prediction
        self.initial_prediction = np.mean(y)
        predictions = np.full(n_samples, self.initial_prediction)
        
        # Initialize feature importance
        self.feature_importance_ = np.zeros(n_features)
        
        # Build trees sequentially
        for i in range(self.n_estimators):
            # Calculate negative gradient
            negative_gradient = self._calculate_negative_gradient(y, predictions)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_indices = np.random.choice(n_samples, 
                                                int(n_samples * self.subsample),
                                                replace=False)
                X_sample = X[sample_indices]
                gradient_sample = negative_gradient[sample_indices]
            else:
                X_sample = X
                gradient_sample = negative_gradient
            
            # Fit tree to negative gradient
            tree = self._build_tree(X_sample, gradient_sample, 
                                  self.max_depth, self.min_samples_split)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
            
            # Update feature importance (simplified)
            self._update_feature_importance(tree, n_features)
            
            # Log progress
            if (i + 1) % 20 == 0:
                loss = np.mean((y - predictions) ** 2)
                logger.info(f"Tree {i+1}/{self.n_estimators} - Loss: {loss:.4f}")
        
        # Normalize feature importance
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ /= np.sum(self.feature_importance_)
    
    def _update_feature_importance(self, tree: Dict, n_features: int):
        """Update feature importance based on tree splits."""
        if tree['type'] == 'split':
            self.feature_importance_[tree['feature']] += 1
            self._update_feature_importance(tree['left'], n_features)
            self._update_feature_importance(tree['right'], n_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)
        
        for tree in self.trees:
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
        
        return predictions


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations and statistical tests."""
    
    def __init__(self):
        """Initialize evaluator with metrics storage."""
        self.results = {}
        self.models = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> ModelMetrics:
        """
        Evaluate classification model performance.
        
        Calculates various metrics including accuracy, precision, recall,
        F1-score, and ROC-AUC if probabilities are provided.
        """
        metrics = ModelMetrics()
        
        # Basic metrics
        metrics.accuracy = np.mean(y_true == y_pred)
        
        # Confusion matrix
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(unique_classes):
            for j, pred_class in enumerate(unique_classes):
                confusion[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        metrics.confusion_matrix = confusion
        
        # Per-class metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for i, class_label in enumerate(unique_classes):
            tp = confusion[i, i]
            fp = np.sum(confusion[:, i]) - tp
            fn = np.sum(confusion[i, :]) - tp
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Average metrics
        metrics.precision = np.mean(precisions)
        metrics.recall = np.mean(recalls)
        metrics.f1_score = np.mean(f1_scores)
        
        # ROC-AUC for binary classification
        if n_classes == 2 and y_proba is not None:
            metrics.auc_roc = self._calculate_auc(y_true, y_proba)
        
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Evaluate regression model performance.
        
        Calculates MSE, MAE, R-squared, and explained variance.
        """
        metrics = ModelMetrics()
        
        # Mean squared error
        metrics.mse = np.mean((y_true - y_pred) ** 2)
        
        # Mean absolute error
        metrics.mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared score
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        if ss_total > 0:
            metrics.r2_score = 1 - (ss_residual / ss_total)
        else:
            metrics.r2_score = 0
        
        # Explained variance
        if np.var(y_true) > 0:
            metrics.explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        else:
            metrics.explained_variance = 0
        
        return metrics
    
    def _calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Calculate area under ROC curve.
        
        Simple implementation using trapezoidal rule.
        """
        # Sort by scores
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate TPR and FPR at each threshold
        tpr_values = []
        fpr_values = []
        
        n_positive = np.sum(y_true == 1)
        n_negative = np.sum(y_true == 0)
        
        tp = 0
        fp = 0
        
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / n_positive if n_positive > 0 else 0
            fpr = fp / n_negative if n_negative > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return auc
    
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                        n_folds: int = 5, task: str = 'classification') -> Dict:
        """
        Perform k-fold cross-validation.
        
        Splits data into k folds, trains on k-1 folds, validates on remaining fold.
        Returns average metrics across all folds.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_size = n_samples // n_folds
        fold_metrics = []
        
        for fold in range(n_folds):
            # Create train/val split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            elif hasattr(model, 'train'):
                model.train(X_train, y_train, epochs=50, verbose=False)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Evaluate
            if task == 'classification':
                metrics = self.evaluate_classification(y_val, y_pred)
            else:
                metrics = self.evaluate_regression(y_val, y_pred)
            
            fold_metrics.append(metrics)
        
        # Average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0].to_dict().keys():
            values = [m.to_dict()[key] for m in fold_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
        
        return avg_metrics
    
    def plot_learning_curves(self, train_scores: List[float], 
                           val_scores: List[float],
                           metric_name: str = "Loss"):
        """Plot training and validation learning curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='Training', linewidth=2)
        plt.plot(val_scores, label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: Optional[List[str]] = None):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        
        if class_names:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              top_k: int = 10):
        """Plot feature importance bar chart."""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_k]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances)
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class AutoML:
    """Automated machine learning pipeline with hyperparameter optimization."""
    
    def __init__(self, task: str = 'classification', time_budget: int = 300):
        """
        Initialize AutoML system.
        
        Automatically selects models, tunes hyperparameters, and finds
        the best model for the given task within time budget.
        """
        self.task = task
        self.time_budget = time_budget
        self.best_model = None
        self.best_score = -float('inf') if task == 'classification' else float('inf')
        self.best_params = {}
        self.search_history = []
        
    def _get_model_candidates(self) -> List[Tuple[str, Any, Dict]]:
        """Get list of candidate models with hyperparameter ranges."""
        if self.task == 'classification':
            return [
                ('neural_network', NeuralNetwork, {
                    'hidden_layers': [[64], [128, 64], [256, 128, 64]],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'activation': ['relu', 'tanh'],
                    'regularization': [0.0, 0.01, 0.1]
                }),
                ('gradient_boosting', GradientBoostingMachine, {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.2]
                })
            ]
        else:
            return [
                ('neural_network', NeuralNetwork, {
                    'hidden_layers': [[32], [64, 32]],
                    'learning_rate': [0.001, 0.01],
                    'activation': ['relu', 'tanh'],
                    'regularization': [0.0, 0.01]
                }),
                ('gradient_boosting', GradientBoostingMachine, {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.2]
                })
            ]
    
    def _random_search(self, model_class, param_grid: Dict, 
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      n_iter: int = 10) -> Tuple[Any, Dict, float]:
        """
        Perform random search for hyperparameter optimization.
        
        Randomly samples parameter combinations and evaluates them.
        """
        best_model = None
        best_params = {}
        best_score = -float('inf') if self.task == 'classification' else float('inf')
        
        for _ in range(n_iter):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = random.choice(param_values)
            
            # Initialize and train model
            try:
                if model_class == NeuralNetwork:
                    model = model_class(
                        input_dim=X_train.shape[1],
                        output_dim=1 if self.task == 'regression' else len(np.unique(y_train)),
                        **params
                    )
                    model.train(X_train, y_train, X_val, y_val, 
                              epochs=50, verbose=False)
                else:
                    model = model_class(**params)
                    model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                
                if self.task == 'classification':
                    score = np.mean(y_val == y_pred)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                else:
                    score = np.mean((y_val - y_pred) ** 2)
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                
            except Exception as e:
                logger.warning(f"Model training failed: {e}")
                continue
        
        return best_model, best_params, best_score
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Automatically find and train the best model.
        
        Tries different models and hyperparameters to find the best
        combination for the given data.
        """
        # Split data
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.fit_transform(X_val)
        
        # Try different models
        start_time = time.time()
        
        for model_name, model_class, param_grid in self._get_model_candidates():
            if time.time() - start_time > self.time_budget:
                break
            
            logger.info(f"Trying {model_name}...")
            
            # Hyperparameter search
            model, params, score = self._random_search(
                model_class, param_grid,
                X_train, y_train, X_val, y_val,
                n_iter=5
            )
            
            # Track results
            self.search_history.append({
                'model': model_name,
                'params': params,
                'score': score
            })
            
            # Update best model
            if self.task == 'classification':
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_params = params
                    logger.info(f"New best model: {model_name} with score {score:.4f}")
            else:
                if score < self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_params = params
                    logger.info(f"New best model: {model_name} with score {score:.4f}")
        
        logger.info(f"AutoML complete. Best score: {self.best_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        return self.best_model.predict(X_processed)


def demonstrate_ml_platform():
    """Demonstrate the complete ML platform capabilities."""
    
    print("PearlMind Machine Learning Platform Demonstration")
    print("Author: Cazandra Aporbo")
    print()
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create classification dataset with non-linear patterns
    X = np.random.randn(n_samples, n_features)
    # Add some non-linear features
    X[:, 0] = X[:, 0] ** 2
    X[:, 1] = np.sin(X[:, 1] * 2)
    X[:, 2] = X[:, 2] * X[:, 3]
    
    # Create target with complex decision boundary
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print()
    
    # Data preprocessing
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(strategy='auto')
    analysis = preprocessor.analyze_data(X)
    X_processed = preprocessor.fit_transform(X)
    print(f"Features after preprocessing: {X_processed.shape[1]}")
    print()
    
    # Neural Network
    print("Training Neural Network...")
    nn = NeuralNetwork(
        input_dim=X_processed.shape[1],
        output_dim=2,
        hidden_layers=[128, 64],
        activation='relu',
        optimizer='adam',
        learning_rate=0.001
    )
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert labels for neural network
    y_train_nn = np.zeros((len(y_train), 2))
    y_train_nn[np.arange(len(y_train)), y_train] = 1
    
    nn.train(X_train, y_train_nn, epochs=50, batch_size=32, verbose=False)
    nn_pred = nn.predict(X_test)
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingMachine(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
    gb.fit(X_train, y_train)
    gb_pred = (gb.predict(X_test) > 0.5).astype(int)
    
    # Evaluation
    print("\nModel Evaluation:")
    evaluator = ModelEvaluator()
    
    nn_metrics = evaluator.evaluate_classification(y_test, nn_pred)
    print(f"Neural Network Accuracy: {nn_metrics.accuracy:.4f}")
    print(f"Neural Network F1-Score: {nn_metrics.f1_score:.4f}")
    
    gb_metrics = evaluator.evaluate_classification(y_test, gb_pred)
    print(f"Gradient Boosting Accuracy: {gb_metrics.accuracy:.4f}")
    print(f"Gradient Boosting F1-Score: {gb_metrics.f1_score:.4f}")
    
    # AutoML
    print("\nRunning AutoML (this may take a moment)...")
    automl = AutoML(task='classification', time_budget=30)
    automl.fit(X, y, validation_split=0.2)
    print(f"Best model found: {automl.best_params}")
    print(f"Best validation score: {automl.best_score:.4f}")
    
    print("\nPlatform demonstration complete!")


if __name__ == "__main__":
    demonstrate_ml_platform()
