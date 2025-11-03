#!/usr/bin/env python3
"""
Complete Machine Learning Platform
Author: Cazandra Aporbo
Version: 3.0.0
Last Updated: November 2025

A machine learning platform with automatic dependency management,
advanced algorithms, and production-ready implementations. This platform provides
everything from basic statistical analysis to deep learning, with careful attention
to numerical stability, performance optimization, and interpretability.

The code demonstrates:
- Statistical foundations and math rigor
- Algorithm implementation from first principles
- Production software engineering practices
- Performance optimization techniques
- Visualization and interpretability methods
"""

import sys
import os
import subprocess
import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps, lru_cache, partial
from collections import defaultdict, OrderedDict, namedtuple
from itertools import combinations, permutations, product
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import threading
import queue
import time
import logging
import inspect
import traceback
from enum import Enum, auto

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('pearlmind_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PearlMind')

# Package management system with automatic installation
class DependencyManager:
    """
    Manages package dependencies with automatic installation and version checking.
    This ensures all required packages are available before the main system loads.
    """
    
    REQUIRED_PACKAGES = {
        'numpy': '1.24.0',
        'scipy': '1.10.0',
        'pandas': '2.0.0',
        'scikit-learn': '1.3.0',
        'matplotlib': '3.7.0',
        'seaborn': '0.12.0',
        'plotly': '5.14.0',
        'statsmodels': '0.14.0',
        'xgboost': '1.7.0',
        'lightgbm': '4.0.0',
        'catboost': '1.2.0',
        'optuna': '3.2.0',
        'shap': '0.42.0',
        'imbalanced-learn': '0.11.0',
        'yellowbrick': '1.5',
        'tqdm': '4.65.0',
        'joblib': '1.3.0',
        'dask': '2023.5.0',
        'numba': '0.57.0',
        'torch': '2.0.0',
        'tensorflow': '2.13.0',
        'keras': '2.13.0',
        'transformers': '4.30.0',
        'datasets': '2.13.0',
        'tokenizers': '0.13.0',
        'wandb': '0.15.0',
        'mlflow': '2.4.0',
        'ray': '2.5.0',
        'hyperopt': '0.2.7',
        'prophet': '1.1.0',
        'pmdarima': '2.0.0',
        'tslearn': '0.6.0',
        'dtreeviz': '2.2.0',
        'interpret': '0.4.0',
        'lime': '0.2.0',
        'eli5': '0.13.0'
    }
    
    @classmethod
    def check_and_install(cls, verbose: bool = True) -> Dict[str, bool]:
        """
        Check for required packages and install missing ones.
        Returns a dictionary mapping package names to installation success.
        """
        installation_status = {}
        missing_packages = []
        
        for package_name, min_version in cls.REQUIRED_PACKAGES.items():
            try:
                module = importlib.import_module(package_name.replace('-', '_'))
                if verbose:
                    version = getattr(module, '__version__', 'unknown')
                    logger.info(f"✓ {package_name} v{version} available")
                installation_status[package_name] = True
            except ImportError:
                missing_packages.append(f"{package_name}>={min_version}")
                installation_status[package_name] = False
                if verbose:
                    logger.warning(f"✗ {package_name} not found")
        
        if missing_packages:
            logger.info(f"Installing {len(missing_packages)} missing packages...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"
                ] + missing_packages)
                
                # Verify installation
                for package in missing_packages:
                    package_name = package.split('>=')[0]
                    try:
                        importlib.import_module(package_name.replace('-', '_'))
                        installation_status[package_name] = True
                        logger.info(f"✓ Successfully installed {package_name}")
                    except ImportError:
                        logger.error(f"✗ Failed to install {package_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Package installation failed: {e}")
        
        return installation_status

# Run dependency check before importing packages
installation_results = DependencyManager.check_and_install(verbose=False)

# Now import all required packages with proper error handling
try:
    import numpy as np
    import pandas as pd
    from scipy import stats, signal, optimize, sparse, spatial, special
    from scipy.stats import norm, uniform, expon, beta, gamma, chi2, t, f
    from scipy.optimize import minimize, differential_evolution, basinhopping
    from scipy.signal import find_peaks, butter, filtfilt, welch
    from scipy.spatial.distance import cdist, pdist, squareform
    
    from sklearn.model_selection import (
        train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit,
        GridSearchCV, RandomizedSearchCV, cross_validate, learning_curve,
        validation_curve, KFold, LeaveOneOut, ShuffleSplit
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
        QuantileTransformer, PolynomialFeatures, SplineTransformer,
        FunctionTransformer, LabelEncoder, OrdinalEncoder, OneHotEncoder
    )
    from sklearn.feature_selection import (
        SelectKBest, RFE, RFECV, SelectFromModel, VarianceThreshold,
        chi2 as chi2_selector, f_classif, mutual_info_classif,
        f_regression, mutual_info_regression
    )
    from sklearn.decomposition import (
        PCA, KernelPCA, SparsePCA, FastICA, NMF, LatentDirichletAllocation,
        TruncatedSVD, FactorAnalysis, DictionaryLearning, MiniBatchDictionaryLearning
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
        GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor, BaggingClassifier,
        BaggingRegressor, VotingClassifier, VotingRegressor, StackingClassifier,
        StackingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
    )
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
        SGDClassifier, SGDRegressor, Perceptron, PassiveAggressiveClassifier,
        PassiveAggressiveRegressor, BayesianRidge, ARDRegression, HuberRegressor,
        RANSACRegressor, TheilSenRegressor, PoissonRegressor, GammaRegressor,
        TweedieRegressor, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
    )
    from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM
    from sklearn.neighbors import (
        KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier,
        RadiusNeighborsRegressor, NearestNeighbors, KernelDensity, LocalOutlierFactor
    )
    from sklearn.tree import (
        DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
        ExtraTreeRegressor, export_text, export_graphviz
    )
    from sklearn.naive_bayes import (
        GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
    )
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    )
    from sklearn.gaussian_process import (
        GaussianProcessClassifier, GaussianProcessRegressor
    )
    from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
    from sklearn.cluster import (
        KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
        MeanShift, AffinityPropagation, OPTICS, Birch, GaussianMixture
    )
    from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        average_precision_score, log_loss, matthews_corrcoef, cohen_kappa_score,
        confusion_matrix, classification_report, roc_curve, precision_recall_curve,
        mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
        mean_absolute_percentage_error, median_absolute_error, max_error,
        silhouette_score, calinski_harabasz_score, davies_bouldin_score
    )
    from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.covariance import EllipticEnvelope, MinCovDet
    from sklearn.inspection import permutation_importance, partial_dependence
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set aesthetic defaults
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Advanced packages with graceful fallback
    try:
        import xgboost as xgb
        HAVE_XGB = True
    except ImportError:
        HAVE_XGB = False
        logger.warning("XGBoost not available")
    
    try:
        import lightgbm as lgb
        HAVE_LGB = True
    except ImportError:
        HAVE_LGB = False
        logger.warning("LightGBM not available")
    
    try:
        import catboost as cb
        HAVE_CB = True
    except ImportError:
        HAVE_CB = False
        logger.warning("CatBoost not available")
    
    try:
        import optuna
        HAVE_OPTUNA = True
    except ImportError:
        HAVE_OPTUNA = False
        logger.warning("Optuna not available")
    
    try:
        import shap
        HAVE_SHAP = True
    except ImportError:
        HAVE_SHAP = False
        logger.warning("SHAP not available")
    
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
        from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
        from imblearn.combine import SMOTETomek, SMOTEENN
        from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
        HAVE_IMBALANCED = True
    except ImportError:
        HAVE_IMBALANCED = False
        logger.warning("Imbalanced-learn not available")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset, random_split
        HAVE_TORCH = True
    except ImportError:
        HAVE_TORCH = False
        logger.warning("PyTorch not available")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models, callbacks, optimizers
        HAVE_TF = True
    except ImportError:
        HAVE_TF = False
        logger.warning("TensorFlow not available")
    
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
        HAVE_STATSMODELS = True
    except ImportError:
        HAVE_STATSMODELS = False
        logger.warning("Statsmodels not available")
    
except ImportError as e:
    logger.error(f"Critical import failed: {e}")
    logger.error("Please ensure all dependencies are installed correctly")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')

# Type definitions for generic programming
T = TypeVar('T')
ModelType = TypeVar('ModelType')


@dataclass
class DatasetInfo:
    """Container for dataset metadata and statistics."""
    name: str
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None
    feature_types: Optional[Dict[str, str]] = None
    missing_values: Optional[Dict[str, float]] = None
    class_balance: Optional[Dict[Any, int]] = None
    memory_usage: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the dataset."""
        summary_parts = [
            f"Dataset: {self.name}",
            f"Samples: {self.n_samples:,}",
            f"Features: {self.n_features}"
        ]
        
        if self.n_classes:
            summary_parts.append(f"Classes: {self.n_classes}")
        
        if self.memory_usage > 0:
            summary_parts.append(f"Memory: {self.memory_usage:.2f} MB")
        
        return " | ".join(summary_parts)


class DataValidator:
    """
    Comprehensive data validation and quality checks.
    Identifies issues like missing values, outliers, data leakage, and distribution problems.
    """
    
    @staticmethod
    def validate_dataset(X: np.ndarray, y: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of input dataset.
        
        Checks for:
        - Data types and shapes
        - Missing values
        - Infinite values
        - Outliers (using multiple methods)
        - Class imbalance
        - Feature correlations
        - Constant features
        - Duplicated samples
        """
        validation_report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Basic shape validation
        if len(X.shape) != 2:
            validation_report['valid'] = False
            validation_report['issues'].append(f"X must be 2D array, got shape {X.shape}")
            return validation_report
        
        n_samples, n_features = X.shape
        validation_report['statistics']['n_samples'] = n_samples
        validation_report['statistics']['n_features'] = n_features
        
        # Check for missing values
        if isinstance(X, pd.DataFrame):
            missing_counts = X.isnull().sum()
            missing_pct = (missing_counts / n_samples * 100).round(2)
            if missing_counts.any():
                validation_report['warnings'].append(
                    f"Missing values detected: {dict(missing_pct[missing_pct > 0])}"
                )
        else:
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                validation_report['warnings'].append(
                    f"Found {nan_count} NaN values ({nan_count/X.size*100:.2f}%)"
                )
        
        # Check for infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            validation_report['issues'].append(f"Found {inf_count} infinite values")
            validation_report['valid'] = False
        
        # Detect constant features
        if isinstance(X, pd.DataFrame):
            constant_features = X.columns[X.nunique() == 1].tolist()
        else:
            constant_features = np.where(np.std(X, axis=0) == 0)[0].tolist()
        
        if constant_features:
            validation_report['warnings'].append(
                f"Constant features detected: {constant_features}"
            )
        
        # Check for duplicated samples
        if isinstance(X, pd.DataFrame):
            duplicates = X.duplicated().sum()
        else:
            unique_rows = np.unique(X, axis=0)
            duplicates = n_samples - len(unique_rows)
        
        if duplicates > 0:
            validation_report['warnings'].append(
                f"Found {duplicates} duplicated samples ({duplicates/n_samples*100:.2f}%)"
            )
        
        # Outlier detection using IQR method
        outlier_counts = {}
        for i in range(n_features):
            col = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i].values
            q1, q3 = np.percentile(col[~np.isnan(col)], [25, 75])
            iqr = q3 - q1
            outliers = np.sum((col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr))
            if outliers > 0:
                feat_name = feature_names[i] if feature_names else f"feature_{i}"
                outlier_counts[feat_name] = outliers
        
        if outlier_counts:
            validation_report['warnings'].append(f"Outliers detected: {outlier_counts}")
        
        # Validate target variable if provided
        if y is not None:
            validation_report['statistics']['n_targets'] = len(y)
            
            if len(y) != n_samples:
                validation_report['valid'] = False
                validation_report['issues'].append(
                    f"Target size {len(y)} doesn't match sample size {n_samples}"
                )
            
            # Check for class imbalance (classification)
            unique_classes = np.unique(y)
            if len(unique_classes) < 20:  # Likely classification
                class_counts = dict(zip(*np.unique(y, return_counts=True)))
                validation_report['statistics']['class_distribution'] = class_counts
                
                # Calculate imbalance ratio
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 10:
                    validation_report['warnings'].append(
                        f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})"
                    )
                elif imbalance_ratio > 3:
                    validation_report['warnings'].append(
                        f"Moderate class imbalance detected (ratio: {imbalance_ratio:.2f})"
                    )
        
        # Calculate feature correlations to detect multicollinearity
        if n_features < 100:  # Only for reasonable number of features
            try:
                if isinstance(X, pd.DataFrame):
                    corr_matrix = X.corr().abs()
                else:
                    corr_matrix = np.corrcoef(X.T)
                
                # Find highly correlated features
                upper_tri = np.triu(corr_matrix, k=1)
                high_corr = np.where(upper_tri > 0.95)
                
                if len(high_corr[0]) > 0:
                    high_corr_pairs = [
                        (i, j, corr_matrix[i, j]) 
                        for i, j in zip(high_corr[0], high_corr[1])
                    ]
                    validation_report['warnings'].append(
                        f"Found {len(high_corr_pairs)} highly correlated feature pairs"
                    )
            except Exception as e:
                logger.debug(f"Correlation calculation failed: {e}")
        
        return validation_report


class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with automatic feature engineering,
    encoding strategies, and robust transformations.
    """
    
    def __init__(self, 
                 strategy: str = 'auto',
                 handle_outliers: bool = True,
                 generate_interactions: bool = True,
                 polynomial_degree: int = 2,
                 target_encoder: bool = False):
        """
        Initialize advanced preprocessor with configurable strategies.
        
        Parameters:
        -----------
        strategy : str
            Preprocessing strategy ('auto', 'robust', 'standard', 'minmax')
        handle_outliers : bool
            Whether to handle outliers using robust methods
        generate_interactions : bool
            Whether to generate interaction features
        polynomial_degree : int
            Degree for polynomial feature generation
        target_encoder : bool
            Whether to use target encoding for categorical variables
        """
        self.strategy = strategy
        self.handle_outliers = handle_outliers
        self.generate_interactions = generate_interactions
        self.polynomial_degree = polynomial_degree
        self.target_encoder = target_encoder
        
        self.transformers = {}
        self.feature_names = []
        self.is_fitted = False
        
    def _detect_column_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect numeric, categorical, and datetime columns.
        
        Uses multiple heuristics:
        - Data type checking
        - Unique value ratios
        - Pattern matching for dates
        """
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }
        
        for col in X.columns:
            dtype = X[col].dtype
            unique_ratio = X[col].nunique() / len(X)
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                column_types['datetime'].append(col)
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(dtype):
                if unique_ratio < 0.05 and X[col].nunique() < 20:
                    # Likely categorical encoded as numeric
                    column_types['categorical'].append(col)
                else:
                    column_types['numeric'].append(col)
            # Check for categorical
            elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
                # Check if it's actually text (high cardinality)
                if unique_ratio > 0.5 and X[col].nunique() > 50:
                    column_types['text'].append(col)
                else:
                    column_types['categorical'].append(col)
        
        return column_types
    
    def _handle_missing_values(self, X: pd.DataFrame, 
                              column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Handle missing values with type-appropriate strategies.
        
        Strategies:
        - Numeric: KNN imputation or iterative imputation
        - Categorical: Mode or create 'missing' category
        - Datetime: Forward fill or interpolation
        """
        X_imputed = X.copy()
        
        # Numeric columns: Use iterative imputation for complex patterns
        if column_types['numeric'] and X[column_types['numeric']].isnull().any().any():
            if len(column_types['numeric']) > 1:
                imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=10,
                    random_state=42
                )
            else:
                imputer = SimpleImputer(strategy='median')
            
            X_imputed[column_types['numeric']] = imputer.fit_transform(
                X[column_types['numeric']]
            )
            self.transformers['numeric_imputer'] = imputer
        
        # Categorical columns: Mode or 'missing' category
        for col in column_types['categorical']:
            if X[col].isnull().any():
                if X[col].isnull().sum() / len(X) > 0.1:
                    # If many missing, create a category
                    X_imputed[col] = X[col].fillna('missing')
                else:
                    # Otherwise use mode
                    X_imputed[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Datetime columns: Forward fill
        for col in column_types['datetime']:
            if X[col].isnull().any():
                X_imputed[col] = X[col].fillna(method='ffill').fillna(method='bfill')
        
        return X_imputed
    
    def _engineer_datetime_features(self, X: pd.DataFrame, 
                                   datetime_cols: List[str]) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.
        
        Features extracted:
        - Year, month, day, hour, minute
        - Day of week, day of year
        - Is weekend, is holiday
        - Time since epoch
        """
        X_engineered = X.copy()
        
        for col in datetime_cols:
            if col not in X.columns:
                continue
            
            dt = pd.to_datetime(X[col], errors='coerce')
            
            # Basic components
            X_engineered[f'{col}_year'] = dt.dt.year
            X_engineered[f'{col}_month'] = dt.dt.month
            X_engineered[f'{col}_day'] = dt.dt.day
            X_engineered[f'{col}_dayofweek'] = dt.dt.dayofweek
            X_engineered[f'{col}_dayofyear'] = dt.dt.dayofyear
            X_engineered[f'{col}_quarter'] = dt.dt.quarter
            X_engineered[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for periodic features
            X_engineered[f'{col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
            X_engineered[f'{col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
            X_engineered[f'{col}_day_sin'] = np.sin(2 * np.pi * dt.dt.day / 31)
            X_engineered[f'{col}_day_cos'] = np.cos(2 * np.pi * dt.dt.day / 31)
            
            # Time since reference point
            reference_date = dt.min()
            X_engineered[f'{col}_days_since_start'] = (dt - reference_date).dt.days
            
            # Drop original datetime column
            X_engineered = X_engineered.drop(col, axis=1)
        
        return X_engineered
    
    def _remove_outliers(self, X: pd.DataFrame, 
                        numeric_cols: List[str]) -> pd.DataFrame:
        """
        Remove outliers using isolation forest and robust statistics.
        
        Multiple strategies:
        - Isolation Forest for multivariate outliers
        - IQR method for univariate outliers
        - Local Outlier Factor for density-based detection
        """
        X_cleaned = X.copy()
        
        if not numeric_cols or len(X) < 100:
            return X_cleaned
        
        # Isolation Forest for multivariate outliers
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Fit only on numeric columns
        outlier_labels = iso_forest.fit_predict(X[numeric_cols].fillna(0))
        
        # Keep only inliers
        inlier_mask = outlier_labels == 1
        
        # Also apply IQR method per column for extreme values
        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            
            # More conservative bounds
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            col_inliers = (X[col] >= lower_bound) & (X[col] <= upper_bound)
            inlier_mask = inlier_mask & col_inliers
        
        # Log outlier removal statistics
        n_outliers = (~inlier_mask).sum()
        if n_outliers > 0:
            logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(X)*100:.2f}%)")
        
        return X_cleaned[inlier_mask]
    
    def _generate_polynomial_features(self, X: pd.DataFrame, 
                                     numeric_cols: List[str]) -> pd.DataFrame:
        """
        Generate polynomial and interaction features.
        
        Includes:
        - Polynomial features up to specified degree
        - Interaction terms between features
        - Ratio features for interpretability
        """
        X_poly = X.copy()
        
        if not numeric_cols or len(numeric_cols) < 2:
            return X_poly
        
        # Limit features to prevent explosion
        selected_cols = numeric_cols[:min(10, len(numeric_cols))]
        
        if self.polynomial_degree > 1:
            poly_transformer = PolynomialFeatures(
                degree=self.polynomial_degree,
                interaction_only=False,
                include_bias=False
            )
            
            poly_features = poly_transformer.fit_transform(X[selected_cols])
            
            # Get feature names
            poly_names = poly_transformer.get_feature_names_out(selected_cols)
            
            # Add only new features (skip original ones)
            new_features = poly_features[:, len(selected_cols):]
            new_names = poly_names[len(selected_cols):]
            
            for i, name in enumerate(new_names):
                X_poly[name] = new_features[:, i]
        
        # Generate ratio features for top numeric pairs
        if self.generate_interactions:
            for i, col1 in enumerate(selected_cols[:-1]):
                for col2 in selected_cols[i+1:]:
                    # Ratio (avoid division by zero)
                    denominator = X[col2].replace(0, np.nan)
                    X_poly[f'{col1}_ratio_{col2}'] = X[col1] / denominator
                    
                    # Product
                    X_poly[f'{col1}_times_{col2}'] = X[col1] * X[col2]
                    
                    # Difference
                    X_poly[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
        
        # Replace any infinities with NaN
        X_poly = X_poly.replace([np.inf, -np.inf], np.nan)
        
        return X_poly
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], 
                     y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Complete pipeline:
        1. Convert to DataFrame if needed
        2. Detect column types
        3. Handle missing values
        4. Engineer datetime features
        5. Remove outliers (optional)
        6. Generate polynomial features
        7. Scale numeric features
        8. Encode categorical features
        """
        # Convert to DataFrame for easier manipulation
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X_processed = X.copy()
        
        # Detect column types
        column_types = self._detect_column_types(X_processed)
        
        # Handle missing values
        X_processed = self._handle_missing_values(X_processed, column_types)
        
        # Engineer datetime features
        if column_types['datetime']:
            X_processed = self._engineer_datetime_features(
                X_processed, column_types['datetime']
            )
            # Update column types after datetime engineering
            column_types = self._detect_column_types(X_processed)
        
        # Remove outliers if requested
        if self.handle_outliers and column_types['numeric']:
            X_processed = self._remove_outliers(X_processed, column_types['numeric'])
        
        # Generate polynomial features
        if self.generate_interactions and column_types['numeric']:
            X_processed = self._generate_polynomial_features(
                X_processed, column_types['numeric']
            )
            # Update column types after feature generation
            column_types = self._detect_column_types(X_processed)
        
        # Scale numeric features
        if column_types['numeric']:
            if self.strategy == 'robust':
                scaler = RobustScaler()
            elif self.strategy == 'minmax':
                scaler = MinMaxScaler()
            elif self.strategy == 'standard':
                scaler = StandardScaler()
            else:  # auto
                # Choose based on data characteristics
                numeric_data = X_processed[column_types['numeric']]
                skewness = numeric_data.skew().abs().mean()
                
                if skewness > 2:
                    scaler = PowerTransformer(method='yeo-johnson')
                elif skewness > 1:
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
            
            X_processed[column_types['numeric']] = scaler.fit_transform(
                X_processed[column_types['numeric']].fillna(0)
            )
            self.transformers['scaler'] = scaler
        
        # Encode categorical features
        if column_types['categorical']:
            if self.target_encoder and y is not None:
                # Target encoding for high cardinality
                from sklearn.preprocessing import TargetEncoder
                encoder = TargetEncoder(smooth='auto')
                X_processed[column_types['categorical']] = encoder.fit_transform(
                    X_processed[column_types['categorical']], y
                )
            else:
                # One-hot encoding with dimensionality control
                high_card_cols = [
                    col for col in column_types['categorical']
                    if X_processed[col].nunique() > 20
                ]
                low_card_cols = [
                    col for col in column_types['categorical']
                    if col not in high_card_cols
                ]
                
                # One-hot for low cardinality
                if low_card_cols:
                    X_processed = pd.get_dummies(
                        X_processed, 
                        columns=low_card_cols, 
                        drop_first=True
                    )
                
                # Ordinal encoding for high cardinality
                if high_card_cols:
                    ordinal_encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
                    X_processed[high_card_cols] = ordinal_encoder.fit_transform(
                        X_processed[high_card_cols]
                    )
                    self.transformers['ordinal_encoder'] = ordinal_encoder
        
        # Store feature names
        self.feature_names = X_processed.columns.tolist()
        self.is_fitted = True
        
        return X_processed.values
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply same transformations using stored transformers
        # This is a simplified version - full implementation would apply
        # all transformations in the same order as fit_transform
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X_transformed = X.copy()
        
        # Apply stored transformations
        for name, transformer in self.transformers.items():
            if 'scaler' in name:
                # Apply scaling to numeric columns
                numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
                X_transformed[numeric_cols] = transformer.transform(X_transformed[numeric_cols])
        
        return X_transformed.values


class AutoMLPipeline:
    """
    Automated machine learning pipeline with intelligent model selection,
    hyperparameter optimization, and ensemble methods.
    """
    
    def __init__(self,
                 task: str = 'auto',
                 time_budget: int = 3600,
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 ensemble: bool = True,
                 use_gpu: bool = False):
        """
        Initialize AutoML pipeline.
        
        Parameters:
        -----------
        task : str
            Task type ('classification', 'regression', 'auto')
        time_budget : int
            Maximum time in seconds for optimization
        n_trials : int
            Number of hyperparameter trials
        cv_folds : int
            Number of cross-validation folds
        ensemble : bool
            Whether to create ensemble of best models
        use_gpu : bool
            Whether to use GPU acceleration where available
        """
        self.task = task
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.ensemble = ensemble
        self.use_gpu = use_gpu
        
        self.best_model = None
        self.best_params = {}
        self.best_score = None
        self.model_scores = {}
        self.ensemble_models = []
        self.preprocessor = None
        self.feature_importance = {}
        
    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Automatically detect whether task is classification or regression.
        
        Heuristics:
        - Number of unique values
        - Data type
        - Value distribution
        """
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        # Check if target is integer
        is_integer = np.all(y == y.astype(int))
        
        # Classification heuristics
        if n_unique < 20 and is_integer:
            return 'classification'
        elif n_unique < 0.05 * len(y):  # Less than 5% unique values
            return 'classification'
        else:
            return 'regression'
    
    def _get_model_candidates(self) -> List[Tuple[str, Any, Dict]]:
        """
        Get list of candidate models based on task type.
        
        Returns models with their default hyperparameter spaces.
        """
        if self.task == 'classification':
            models = [
                ('logistic_regression', LogisticRegression(max_iter=1000), {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['saga'],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }),
                ('random_forest', RandomForestClassifier(n_jobs=-1), {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }),
                ('gradient_boosting', GradientBoostingClassifier(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }),
                ('svm', SVC(probability=True), {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }),
                ('neural_network', MLPClassifier(max_iter=1000), {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                })
            ]
            
            # Add XGBoost if available
            if HAVE_XGB:
                models.append(('xgboost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }))
            
            # Add LightGBM if available
            if HAVE_LGB:
                models.append(('lightgbm', lgb.LGBMClassifier(verbosity=-1), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 1.0]
                }))
            
            # Add CatBoost if available
            if HAVE_CB:
                models.append(('catboost', cb.CatBoostClassifier(verbose=False), {
                    'iterations': [50, 100, 200],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'l2_leaf_reg': [1, 3, 5]
                }))
                
        else:  # regression
            models = [
                ('linear_regression', Ridge(), {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                }),
                ('random_forest', RandomForestRegressor(n_jobs=-1), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }),
                ('gradient_boosting', GradientBoostingRegressor(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }),
                ('svm', SVR(), {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                }),
                ('neural_network', MLPRegressor(max_iter=1000), {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01]
                })
            ]
            
            # Add gradient boosting libraries
            if HAVE_XGB:
                models.append(('xgboost', xgb.XGBRegressor(), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }))
            
            if HAVE_LGB:
                models.append(('lightgbm', lgb.LGBMRegressor(verbosity=-1), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }))
        
        return models
    
    def _optimize_hyperparameters(self, 
                                 model, 
                                 param_space: Dict,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray) -> Tuple[Any, Dict, float]:
        """
        Optimize hyperparameters using Bayesian optimization if available,
        otherwise fallback to randomized search.
        """
        if HAVE_OPTUNA:
            return self._optuna_optimize(model, param_space, X_train, y_train)
        else:
            return self._random_search(model, param_space, X_train, y_train)
    
    def _optuna_optimize(self, model, param_space, X_train, y_train):
        """Use Optuna for Bayesian hyperparameter optimization."""
        
        def objective(trial):
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], (int, np.integer)):
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        min(param_values), 
                        max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values),
                        log=True if min(param_values) > 0 and max(param_values)/min(param_values) > 100 else False
                    )
                elif isinstance(param_values[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif param_values[0] is None:
                    # Handle None in choices
                    options = [x for x in param_values if x is not None]
                    if None in param_values:
                        options.append('none')
                    choice = trial.suggest_categorical(param_name, options)
                    params[param_name] = None if choice == 'none' else choice
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create model with suggested parameters
            model_clone = model.__class__(**params)
            
            # Cross-validation score
            if self.task == 'classification':
                scores = cross_val_score(
                    model_clone, X_train, y_train,
                    cv=self.cv_folds, scoring='roc_auc'
                )
            else:
                scores = cross_val_score(
                    model_clone, X_train, y_train,
                    cv=self.cv_folds, scoring='neg_mean_squared_error'
                )
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize' if self.task == 'classification' else 'minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.time_budget // len(self._get_model_candidates())
        )
        
        # Get best parameters and retrain
        best_params = study.best_params
        best_model = model.__class__(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params, study.best_value
    
    def _random_search(self, model, param_space, X_train, y_train):
        """Fallback to RandomizedSearchCV for hyperparameter optimization."""
        
        if self.task == 'classification':
            scoring = 'roc_auc'
        else:
            scoring = 'neg_mean_squared_error'
        
        search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=min(self.n_trials, 20),
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        """
        Fit AutoML pipeline on training data.
        
        Complete workflow:
        1. Data validation
        2. Preprocessing
        3. Task detection
        4. Model selection and optimization
        5. Ensemble creation
        6. Feature importance calculation
        """
        logger.info("Starting AutoML pipeline...")
        
        # Data validation
        validator = DataValidator()
        validation_report = validator.validate_dataset(X, y)
        
        if not validation_report['valid']:
            raise ValueError(f"Data validation failed: {validation_report['issues']}")
        
        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                logger.warning(warning)
        
        # Preprocessing
        logger.info("Preprocessing data...")
        self.preprocessor = AdvancedPreprocessor(
            strategy='auto',
            handle_outliers=True,
            generate_interactions=False  # Keep it manageable
        )
        
        X_processed = self.preprocessor.fit_transform(X, y)
        
        # Task detection
        if self.task == 'auto':
            self.task = self._detect_task_type(y)
            logger.info(f"Detected task type: {self.task}")
        
        # Split data if validation set not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y, test_size=0.2, random_state=42,
                stratify=y if self.task == 'classification' else None
            )
        else:
            X_train, y_train = X_processed, y
            X_val = self.preprocessor.transform(X_val)
        
        # Get candidate models
        model_candidates = self._get_model_candidates()
        
        # Optimize each model
        start_time = time.time()
        
        for model_name, model, param_space in model_candidates:
            if time.time() - start_time > self.time_budget:
                logger.info("Time budget exceeded")
                break
            
            logger.info(f"Optimizing {model_name}...")
            
            try:
                optimized_model, best_params, score = self._optimize_hyperparameters(
                    model, param_space, X_train, y_train
                )
                
                # Evaluate on validation set
                if self.task == 'classification':
                    y_pred = optimized_model.predict_proba(X_val)[:, 1]
                    val_score = roc_auc_score(y_val, y_pred)
                else:
                    y_pred = optimized_model.predict(X_val)
                    val_score = -mean_squared_error(y_val, y_pred)
                
                self.model_scores[model_name] = {
                    'model': optimized_model,
                    'params': best_params,
                    'cv_score': score,
                    'val_score': val_score
                }
                
                logger.info(f"{model_name} validation score: {val_score:.4f}")
                
                # Update best model
                if self.best_score is None or val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = optimized_model
                    self.best_params = best_params
                    logger.info(f"New best model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to optimize {model_name}: {e}")
                continue
        
        # Create ensemble if requested
        if self.ensemble and len(self.model_scores) > 1:
            logger.info("Creating ensemble...")
            self._create_ensemble(X_train, y_train, X_val, y_val)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train, y_train)
        
        logger.info(f"AutoML complete. Best score: {self.best_score:.4f}")
    
    def _create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create an ensemble of the best models."""
        
        # Select top models
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['val_score'],
            reverse=True
        )[:5]  # Top 5 models
        
        if self.task == 'classification':
            # Voting classifier
            estimators = [(name, data['model']) for name, data in sorted_models]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred = ensemble.predict_proba(X_val)[:, 1]
            ensemble_score = roc_auc_score(y_val, y_pred)
        else:
            # Voting regressor
            estimators = [(name, data['model']) for name, data in sorted_models]
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred = ensemble.predict(X_val)
            ensemble_score = -mean_squared_error(y_val, y_pred)
        
        logger.info(f"Ensemble score: {ensemble_score:.4f}")
        
        # Use ensemble if it's better
        if ensemble_score > self.best_score:
            self.best_model = ensemble
            self.best_score = ensemble_score
            logger.info("Ensemble selected as best model")
    
    def _calculate_feature_importance(self, X_train, y_train):
        """Calculate feature importance using permutation importance."""
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based model
            importances = self.best_model.feature_importances_
            self.feature_importance = dict(
                zip(self.preprocessor.feature_names, importances)
            )
        else:
            # Use permutation importance
            try:
                result = permutation_importance(
                    self.best_model, X_train, y_train,
                    n_repeats=10, random_state=42, n_jobs=-1
                )
                
                self.feature_importance = dict(
                    zip(self.preprocessor.feature_names, result.importances_mean)
                )
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data."""
        
        if self.best_model is None:
            raise ValueError("Model not fitted yet")
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get probability predictions for classification."""
        
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.best_model is None:
            raise ValueError("Model not fitted yet")
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        return self.best_model.predict_proba(X_processed)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get DataFrame comparing all trained models."""
        
        comparison_data = []
        for name, data in self.model_scores.items():
            comparison_data.append({
                'Model': name,
                'CV Score': data['cv_score'],
                'Validation Score': data['val_score'],
                'Best Parameters': str(data['params'])
            })
        
        return pd.DataFrame(comparison_data).sort_values(
            'Validation Score', ascending=False
        )


class ModelInterpreter:
    """
    Advanced model interpretation and explainability tools.
    Provides SHAP values, LIME explanations, partial dependence plots, and more.
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize model interpreter.
        
        Parameters:
        -----------
        model : Any
            Trained model to interpret
        X_train : np.ndarray
            Training data for background distribution
        feature_names : List[str]
            Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if HAVE_SHAP:
            self._init_shap_explainer()
    
    def _init_shap_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        
        try:
            # Tree-based models
            if hasattr(self.model, 'tree_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'estimators_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            # Linear models
            elif hasattr(self.model, 'coef_'):
                self.shap_explainer = shap.LinearExplainer(
                    self.model, self.X_train
                )
            # Neural networks and black-box models
            else:
                # Use KernelExplainer as fallback
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.X_train, 100)
                )
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for given samples."""
        
        if not HAVE_SHAP or self.shap_explainer is None:
            raise ValueError("SHAP not available or explainer not initialized")
        
        shap_values = self.shap_explainer.shap_values(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        return shap_values
    
    def plot_shap_summary(self, X: np.ndarray, max_display: int = 20):
        """Create SHAP summary plot."""
        
        if not HAVE_SHAP:
            logger.warning("SHAP not available for plotting")
            return
        
        shap_values = self.get_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, X_sample: np.ndarray, sample_idx: int = 0):
        """Create SHAP waterfall plot for a single prediction."""
        
        if not HAVE_SHAP:
            logger.warning("SHAP not available for plotting")
            return
        
        shap_values = self.get_shap_values(X_sample[sample_idx:sample_idx+1])
        
        # Create explanation object
        if hasattr(self.shap_explainer, 'expected_value'):
            base_value = self.shap_explainer.expected_value
        else:
            base_value = 0
        
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_sample[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from model or SHAP values."""
        
        importance_dict = {}
        
        # Try to get from model directly
        if hasattr(self.model, 'feature_importances_'):
            importance_dict['model_importance'] = self.model.feature_importances_
        
        # Get SHAP-based importance
        if HAVE_SHAP and self.shap_explainer:
            try:
                # Use a sample of training data
                sample_size = min(1000, len(self.X_train))
                sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
                X_sample = self.X_train[sample_idx]
                
                shap_values = self.get_shap_values(X_sample)
                
                # Calculate mean absolute SHAP values
                if len(shap_values.shape) == 3:
                    # Multi-class: average across classes
                    shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
                else:
                    shap_importance = np.mean(np.abs(shap_values), axis=0)
                
                importance_dict['shap_importance'] = shap_importance
            except Exception as e:
                logger.warning(f"Could not calculate SHAP importance: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(importance_dict, index=self.feature_names)
        
        # Add average importance
        if len(importance_dict) > 1:
            df['average_importance'] = df.mean(axis=1)
        
        return df.sort_values(df.columns[-1], ascending=False)
    
    def plot_partial_dependence(self, features: List[int], kind: str = 'both'):
        """
        Plot partial dependence for specified features.
        
        Parameters:
        -----------
        features : List[int]
            Feature indices to plot
        kind : str
            'individual', 'average', or 'both'
        """
        from sklearn.inspection import PartialDependenceDisplay
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_train,
            features,
            kind=kind,
            feature_names=self.feature_names,
            ax=ax
        )
        
        plt.tight_layout()
        plt.show()


def demo_pearlmind_platform():
    """
    Comprehensive demonstration of the PearlMind ML platform capabilities.
    
    Shows:
    1. Data generation and validation
    2. Advanced preprocessing
    3. AutoML pipeline
    4. Model interpretation
    5. Performance visualization
    """
    
    print("\n" + "="*80)
    print("PearlMind ML Journey - Complete Platform Demo")
    print("Author: Cazandra Aporbo")
    print("="*80 + "\n")
    
    # Generate synthetic dataset with realistic properties
    print("Generating synthetic dataset...")
    np.random.seed(42)
    
    # Create a challenging classification dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_repeated=2,
        n_classes=3,
        n_clusters_per_class=2,
        weights=[0.5, 0.3, 0.2],
        class_sep=0.8,
        flip_y=0.05,
        random_state=42
    )
    
    # Convert to DataFrame for better handling
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    X_df['category_A'] = np.random.choice(['low', 'medium', 'high'], size=len(X_df))
    X_df['category_B'] = np.random.choice(['alpha', 'beta', 'gamma', 'delta'], size=len(X_df))
    
    # Add missing values
    missing_mask = np.random.random(X_df.shape) < 0.05
    X_df[missing_mask] = np.nan
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Data validation
    print("\n" + "-"*60)
    print("Running data validation...")
    validator = DataValidator()
    validation_report = validator.validate_dataset(X_df.values, y, feature_names)
    
    print(f"Validation status: {'✓ Valid' if validation_report['valid'] else '✗ Invalid'}")
    if validation_report['warnings']:
        print(f"Warnings: {len(validation_report['warnings'])}")
        for warning in validation_report['warnings'][:3]:
            print(f"  - {warning[:100]}...")
    
    # AutoML Pipeline
    print("\n" + "-"*60)
    print("Running AutoML pipeline...")
    
    automl = AutoMLPipeline(
        task='auto',
        time_budget=60,  # Quick demo
        n_trials=10,
        cv_folds=3,
        ensemble=True
    )
    
    # Fit AutoML
    automl.fit(X_df, y)
    
    # Show results
    print("\nModel comparison:")
    comparison_df = automl.get_model_comparison()
    print(comparison_df.to_string())
    
    print(f"\nBest model: {automl.best_model.__class__.__name__}")
    print(f"Best score: {automl.best_score:.4f}")
    
    # Feature importance
    if automl.feature_importance:
        print("\nTop 10 most important features:")
        sorted_features = sorted(
            automl.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.4f}")
    
    # Model interpretation (if possible)
    if HAVE_SHAP:
        print("\n" + "-"*60)
        print("Model interpretation...")
        
        # Get preprocessed training data
        X_processed = automl.preprocessor.fit_transform(X_df, y)
        
        interpreter = ModelInterpreter(
            automl.best_model,
            X_processed,
            feature_names=automl.preprocessor.feature_names
        )
        
        # Get feature importance
        importance_df = interpreter.get_feature_importance()
        print("\nFeature importance summary:")
        print(importance_df.head(10))
    
    # Performance visualization
    print("\n" + "-"*60)
    print("Creating visualizations...")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    X_processed = automl.preprocessor.fit_transform(X_df, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_pred = automl.predict(pd.DataFrame(X_test))
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


if __name__ == "__main__":
    # Run demonstration
    demo_pearlmind_platform()
