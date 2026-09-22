#!/bin/bash

# PearlMind ML Journey - Project Initialization Script
# Author: Cazandra Aporbo (becaziam@gmail.com)

echo "🚀 Initializing PearlMind ML Journey Project Structure..."

# Create main source directories
echo "Creating source directories..."
mkdir -p src/pearlmind/{data,features,models,evaluation,deployment,utils,cli,scripts}
mkdir -p src/pearlmind/models/{baseline,ensemble,neural,transformers}
mkdir -p src/pearlmind/data/{loaders,processors,validators,splitters}
mkdir -p src/pearlmind/features/{extractors,transformers,store}
mkdir -p src/pearlmind/evaluation/{metrics,fairness,calibration,monitoring}
mkdir -p src/pearlmind/deployment/{serving,preprocessing,postprocessing,monitoring}
mkdir -p src/pearlmind/utils/{config,logging,profiling,visualization}

# Create test directories
echo "Creating test directories..."
mkdir -p tests/{unit,integration,inference,fixtures,benchmarks}
mkdir -p tests/unit/{data,features,models,evaluation,deployment,utils}
mkdir -p tests/fixtures/{data,models,configs}

# Create configuration directories
echo "Creating configuration directories..."
mkdir -p configs/{models,features,deployment,monitoring,experiments}

# Create notebook directories
echo "Creating notebook directories..."
mkdir -p notebooks/{exploration,modeling,evaluation,reports,tutorials}

# Create script directories
echo "Creating script directories..."
mkdir -p scripts/{training,evaluation,deployment,monitoring,data}

# Create documentation directories
echo "Creating documentation directories..."
mkdir -p docs/{api,guides,model_cards,decisions,papers}

# Create data directories
echo "Creating data directories..."
mkdir -p data/{raw,processed,features,cache,external}

# Create model directories
echo "Creating model directories..."
mkdir -p models/{baseline,experiments,production,registry,checkpoints}

# Create logs and outputs
echo "Creating logs and output directories..."
mkdir -p logs/{training,evaluation,serving,monitoring}
mkdir -p outputs/{figures,reports,predictions,exports}

# Create requirements directory
echo "Creating requirements directory..."
mkdir -p requirements

# Create GitHub directories
echo "Creating GitHub directories..."
mkdir -p .github/{workflows,ISSUE_TEMPLATE}

# Create __init__.py files for all Python packages
echo "Creating __init__.py files..."

# Main package init
cat > src/pearlmind/__init__.py << 'EOF'
"""
PearlMind ML Journey
From Mathematical Foundations to Ethical Superintelligence

Author: Cazandra Aporbo
Email: becaziam@gmail.com
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Cazandra Aporbo"
__email__ = "becaziam@gmail.com"

# Import core modules
from pearlmind.utils.config import Config
from pearlmind.utils.logging import get_logger

# Set up logging
logger = get_logger(__name__)

# Package metadata
__all__ = [
    "Config",
    "get_logger",
    "__version__",
    "__author__",
    "__email__",
]
EOF

# Data module init
cat > src/pearlmind/data/__init__.py << 'EOF'
"""Data processing and loading utilities."""

from pearlmind.data.loaders import DataLoader
from pearlmind.data.processors import DataProcessor
from pearlmind.data.validators import DataValidator
from pearlmind.data.splitters import DataSplitter

__all__ = [
    "DataLoader",
    "DataProcessor", 
    "DataValidator",
    "DataSplitter",
]
EOF

# Models module init
cat > src/pearlmind/models/__init__.py << 'EOF'
"""Machine learning models with fairness auditing."""

from pearlmind.models.baseline import BaselineModel
from pearlmind.models.ensemble import EnsembleModel
from pearlmind.models.neural import NeuralModel

__all__ = [
    "BaselineModel",
    "EnsembleModel",
    "NeuralModel",
]
EOF

# Evaluation module init
cat > src/pearlmind/evaluation/__init__.py << 'EOF'
"""Model evaluation and fairness auditing."""

from pearlmind.evaluation.metrics import Metrics
from pearlmind.evaluation.fairness import FairnessAuditor
from pearlmind.evaluation.calibration import CalibrationAnalyzer
from pearlmind.evaluation.monitoring import ModelMonitor

__all__ = [
    "Metrics",
    "FairnessAuditor",
    "CalibrationAnalyzer",
    "ModelMonitor",
]
EOF

# Utils module init
cat > src/pearlmind/utils/__init__.py << 'EOF'
"""Utility functions and helpers."""

from pearlmind.utils.config import Config
from pearlmind.utils.logging import get_logger

__all__ = [
    "Config",
    "get_logger",
]
EOF

# Features module init
cat > src/pearlmind/features/__init__.py << 'EOF'
"""Feature engineering and transformation."""

__all__ = []
EOF

# Deployment module init
cat > src/pearlmind/deployment/__init__.py << 'EOF'
"""Model deployment and serving."""

__all__ = []
EOF

# CLI module init
cat > src/pearlmind/cli/__init__.py << 'EOF'
"""Command-line interface for PearlMind."""

from pearlmind.cli.main import app

__all__ = ["app"]
EOF

# Scripts module init
cat > src/pearlmind/scripts/__init__.py << 'EOF'
"""Executable scripts for training and evaluation."""

__all__ = []
EOF

# Create subdirectory __init__ files
for dir in src/pearlmind/*/; do
    for subdir in "$dir"*/; do
        if [ -d "$subdir" ]; then
            touch "${subdir}__init__.py"
        fi
    done
done

# Create test __init__ files
touch tests/__init__.py
for dir in tests/*/; do
    if [ -d "$dir" ]; then
        touch "${dir}__init__.py"
    fi
done

# Create core configuration file
echo "Creating core configuration..."
cat > src/pearlmind/utils/config.py << 'EOF'
"""Configuration management for PearlMind."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration."""
    
    name: str = "baseline"
    version: str = "1.0.0"
    params: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_dir: Path = Path("models/checkpoints")
    

class TrainingConfig(BaseModel):
    """Training configuration."""
    
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cpu"
    seed: int = 42
    

class FairnessConfig(BaseModel):
    """Fairness audit configuration."""
    
    enabled: bool = True
    metrics: list = Field(default_factory=lambda: ["demographic_parity", "equalized_odds"])
    protected_attributes: list = Field(default_factory=list)
    threshold: float = 0.8
    

class Config(BaseSettings):
    """Main configuration class."""
    
    # Project settings
    project_name: str = "PearlMind ML Journey"
    version: str = "2.0.0"
    environment: str = "development"
    
    # Paths
    data_path: Path = Path("data")
    model_path: Path = Path("models")
    log_path: Path = Path("logs")
    output_path: Path = Path("outputs")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    
    class Config:
        env_prefix = "PEARLMIND_"
        env_file = ".env"
        
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
        
    def save_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Global config instance
config = Config()
EOF

# Create logging utility
echo "Creating logging utility..."
cat > src/pearlmind/utils/logging.py << 'EOF'
"""Logging configuration for PearlMind."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


# Rich console for pretty printing
console = Console()


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True
) -> logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        use_rich: Use rich formatting
        
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()
    
    # Add console handler
    if use_rich:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=level,
            colorize=True,
        )
    else:
        logger.add(sys.stderr, level=level)
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )
    
    return logger


# Create default logger
default_logger = get_logger("pearlmind")
EOF

# Create base model class
echo "Creating base model class..."
cat > src/pearlmind/models/base.py << 'EOF'
"""Base model class with fairness auditing capabilities."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from pearlmind.evaluation.fairness import FairnessAuditor
from pearlmind.utils.logging import get_logger


logger = get_logger(__name__)


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all models with fairness auditing.
    
    All models in PearlMind must inherit from this class and implement
    the required methods for training, prediction, and fairness auditing.
    """
    
    def __init__(
        self,
        name: str = "base_model",
        version: str = "1.0.0",
        enable_fairness_audit: bool = True,
        **kwargs
    ):
        """
        Initialize base model.
        
        Args:
            name: Model name
            version: Model version
            enable_fairness_audit: Enable automatic fairness auditing
            **kwargs: Additional model parameters
        """
        self.name = name
        self.version = version
        self.enable_fairness_audit = enable_fairness_audit
        self.params = kwargs
        self.is_fitted = False
        self._fairness_auditor = FairnessAuditor() if enable_fairness_audit else None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training arguments
            
        Returns:
            Fitted model instance
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        pass
        
    def audit_fairness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform fairness audit on the model.
        
        Args:
            X: Features
            y: True labels
            sensitive_features: Protected attributes
            **kwargs: Additional audit parameters
            
        Returns:
            Fairness audit report
        """
        if not self.enable_fairness_audit:
            logger.warning("Fairness audit is disabled for this model")
            return {}
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before auditing")
            
        logger.info(f"Running fairness audit for {self.name}")
        predictions = self.predict(X)
        
        return self._fairness_auditor.audit(
            y_true=y,
            y_pred=predictions,
            sensitive_features=sensitive_features,
            **kwargs
        )
        
    def save(self, path: Path) -> None:
        """Save model to disk."""
        raise NotImplementedError
        
    def load(self, path: Path) -> None:
        """Load model from disk."""
        raise NotImplementedError
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params
        
    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        self.params.update(params)
        return self
        
    def __repr__(self) -> str:
        return f"{self.name}(version={self.version}, fitted={self.is_fitted})"
EOF

# Create fairness auditor stub
echo "Creating fairness auditor..."
cat > src/pearlmind/evaluation/fairness.py << 'EOF'
"""Fairness auditing for ML models."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from pearlmind.utils.logging import get_logger


logger = get_logger(__name__)


class FairnessAuditor:
    """Audit models for bias and fairness."""
    
    def __init__(self):
        """Initialize fairness auditor."""
        self.metrics = {}
        
    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform fairness audit.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_features: Protected attributes
            **kwargs: Additional parameters
            
        Returns:
            Audit report
        """
        report = {
            "overall_accuracy": accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        
        if sensitive_features is not None:
            # Placeholder for actual fairness metrics
            report["fairness_metrics"] = {
                "demographic_parity": self._calculate_demographic_parity(
                    y_true, y_pred, sensitive_features
                ),
                "equalized_odds": self._calculate_equalized_odds(
                    y_true, y_pred, sensitive_features
                ),
            }
            
        return report
        
    def _calculate_demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray
    ) -> float:
        """Calculate demographic parity metric."""
        # Placeholder implementation
        return 0.0
        
    def _calculate_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray
    ) -> float:
        """Calculate equalized odds metric."""
        # Placeholder implementation
        return 0.0
EOF

# Create sample configuration files
echo "Creating sample configuration files..."

cat > configs/default.yaml << 'EOF'
# Default configuration for PearlMind ML Journey
project_name: "PearlMind ML Journey"
version: "2.0.0"
environment: "development"

# Model configuration
model:
  name: "baseline"
  version: "1.0.0"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

# Training configuration  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  device: "cpu"
  seed: 42

# Fairness configuration
fairness:
  enabled: true
  metrics:
    - demographic_parity
    - equalized_odds
    - calibration
  protected_attributes:
    - gender
    - race
    - age_group
  threshold: 0.8
EOF

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.hypothesis/

# ML/Data
*.h5
*.pkl
*.pickle
*.joblib
*.pt
*.pth
*.onnx
data/raw/*
data/processed/*
models/checkpoints/*
models/experiments/*
logs/*
outputs/*

# Documentation
docs/_build/
docs/_static/
docs/_templates/

# Environment
.env
.env.local
.env.*.local

# OS
Thumbs.db
.DS_Store
EOF

# Create README for src directory
echo "Creating src/README.md..."
cat > src/README.md << 'EOF'
# PearlMind ML Journey - Source Code

This directory contains the main source code for the PearlMind ML Journey project.

## Structure

- `pearlmind/` - Main package directory
  - `data/` - Data loading and processing
  - `features/` - Feature engineering
  - `models/` - ML model implementations
  - `evaluation/` - Model evaluation and fairness auditing
  - `deployment/` - Model serving and deployment
  - `utils/` - Utility functions
  - `cli/` - Command-line interface
  - `scripts/` - Executable scripts

## Development

To work on the source code:

1. Install in editable mode: `pip install -e .`
2. Make your changes
3. Run tests: `pytest tests/`
4. Check code quality: `make lint`
EOF

echo "✅ Project structure initialized successfully!"
echo ""
echo "Next steps:"
echo "1. Create virtual environment: python -m venv .venv"
echo "2. Activate it: source .venv/bin/activate"
echo "3. Install package: pip install -e ."
echo "4. Run tests: pytest tests/"
echo ""
echo "Project structure created at: $(pwd)"
