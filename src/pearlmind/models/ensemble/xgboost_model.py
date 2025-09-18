# src/pearlmind/models/ensemble/xgboost_model.py
"""XGBoost model implementation with fairness auditing."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from pearlmind.models.base import BaseModel
from pearlmind.utils.logging import get_logger


logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model with integrated fairness auditing.
    
    This model wraps XGBoost with additional capabilities for:
    - Automatic fairness auditing
    - Early stopping
    - Feature importance analysis
    - Production optimization
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        objective: str = "binary:logistic",
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        enable_fairness_audit: bool = True,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            objective: Learning objective
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            enable_fairness_audit: Enable automatic fairness auditing
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(
            name="XGBoostModel",
            version="1.0.0",
            enable_fairness_audit=enable_fairness_audit
        )
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        # Store additional parameters
        self.params.update(kwargs)
        
        # Initialize model
        self._model = None
        self._feature_importance = None
        self._training_history = {}
        
    def _get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'verbosity': 1 if self.verbose else 0,
            'use_label_encoder': False,
            'eval_metric': 'logloss' if 'binary' in self.objective else 'mlogloss',
        }
        
        # Add additional parameters
        params.update(self.params)
        
        return params
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[list] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: List of (X, y) tuples for evaluation
            sample_weight: Sample weights
            **kwargs: Additional fit parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Initialize XGBoost model
        self._model = xgb.XGBClassifier(**self._get_xgb_params())
        
        # Prepare fit parameters
        fit_params = {
            'sample_weight': sample_weight,
            'verbose': self.verbose,
        }
        
        # Add evaluation set if provided
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            
        # Update with additional parameters
        fit_params.update(kwargs)
        
        # Train model
        self._model.fit(X, y, **fit_params)
        
        # Extract feature importance
        self._feature_importance = self._model.feature_importances_
        
        # Store training history
        if hasattr(self._model, 'evals_result_'):
            self._training_history = self._model.evals_result_
            
        # Log training metrics
        train_score = accuracy_score(y, self._model.predict(X))
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Run cross-validation
        if len(X) > 100:  # Only for sufficient data
            cv_scores = cross_val_score(self._model, X, y, cv=5)
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        self.is_fitted = True
        
        # Perform fairness audit if enabled
        if self.enable_fairness_audit and 'sensitive_features' in kwargs:
            audit_report = self.audit_fairness(
                X, y, kwargs['sensitive_features']
            )
            logger.info(f"Fairness audit complete: {audit_report}")
            
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self._model.predict(X)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self._model.predict_proba(X)
        
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            feature_names: Optional feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
            
        importance = self._model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
        df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        
        return df.sort_values('importance', ascending=False)
        
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, list],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and scores
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("Starting hyperparameter optimization")
        
        # Create base model
        base_model = xgb.XGBClassifier(**self._get_xgb_params())
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self._model.save_model(str(path.with_suffix('.json')))
        
        # Save additional metadata
        import pickle
        metadata = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'feature_importance': self._feature_importance,
            'training_history': self._training_history,
        }
        
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        # Load XGBoost model
        self._model = xgb.XGBClassifier(**self._get_xgb_params())
        self._model.load_model(str(path.with_suffix('.json')))
        
        # Load metadata
        import pickle
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        self.name = metadata['name']
        self.version = metadata['version']
        self.params = metadata['params']
        self._feature_importance = metadata['feature_importance']
        self._training_history = metadata['training_history']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        
    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        index: int = 0,
        use_shap: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X: Input features
            index: Index of sample to explain
            use_shap: Use SHAP for explanation
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")
            
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Get prediction
        sample = X[index:index+1]
        prediction = self.predict(sample)[0]
        proba = self.predict_proba(sample)[0]
        
        explanation = {
            'prediction': prediction,
            'probability': proba.tolist(),
            'feature_values': dict(zip(feature_names, X[index]))
        }
        
        if use_shap:
            try:
                import shap
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(self._model)
                shap_values = explainer.shap_values(sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Binary classification
                    
                explanation['shap_values'] = dict(zip(
                    feature_names,
                    shap_values[0]
                ))
                
            except ImportError:
                logger.warning("SHAP not installed, using feature importance instead")
                explanation['feature_importance'] = dict(zip(
                    feature_names,
                    self._feature_importance
                ))
                
        return explanation
