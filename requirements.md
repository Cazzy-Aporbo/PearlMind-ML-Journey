# Requirements Files Structure

## requirements/base.txt
Core dependencies for basic functionality:

```txt
# Core Scientific Computing
numpy>=1.23.0,<2.0.0
scipy>=1.10.0
pandas>=2.0.0

# Basic ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.66.0
loguru>=0.7.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

## requirements/pytorch.txt
PyTorch ecosystem:

```txt
-r base.txt

# PyTorch (CPU version - for GPU use torch+cu118)
--index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# PyTorch extras
torchmetrics>=1.2.0
pytorch-lightning>=2.1.0
einops>=0.7.0
timm>=0.9.0
```

## requirements/tensorflow.txt
TensorFlow ecosystem:

```txt
-r base.txt

# TensorFlow
tensorflow>=2.14.0
tensorflow-probability>=0.22.0
tensorflow-addons>=0.22.0
tensorflow-datasets>=4.9.0
```

## requirements/transformers.txt
NLP and Transformers:

```txt
-r base.txt

# Hugging Face ecosystem
transformers>=4.35.0
tokenizers>=0.15.0
datasets>=2.15.0
sentence-transformers>=2.2.0
accelerate>=0.25.0
```

## requirements/dev.txt
Development dependencies:

```txt
-r base.txt

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-xdist>=3.3.0
pytest-benchmark>=4.0.0
pytest-mock>=3.11.0
hypothesis>=6.90.0

# Code Quality
black>=23.0.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
pylint>=3.0.0
bandit>=1.7.0
pre-commit>=3.5.0

# Notebooks
jupyter>=1.0.0
jupyterlab>=4.0.0
nbstripout>=0.6.0

# Documentation
sphinx>=7.2.0
sphinx-rtd-theme>=1.3.0
myst-parser>=2.0.0
```

## requirements/production.txt
Production deployment:

```txt
-r base.txt

# API
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0

# Model Serving
onnx>=1.15.0
onnxruntime>=1.16.0
bentoml>=1.1.0

# Monitoring
prometheus-client>=0.19.0
mlflow>=2.8.0
```

## requirements/fairness.txt
Fairness and ethics tools:

```txt
-r base.txt

# Fairness libraries
fairlearn>=0.10.0
aif360>=0.5.0
aequitas>=2.0.0

# Interpretability
shap>=0.43.0
lime>=0.2.0
captum>=0.7.0  # For PyTorch
```

## requirements/security.txt
Security and adversarial testing:

```txt
-r base.txt

# Adversarial robustness
adversarial-robustness-toolbox>=1.16.0
foolbox>=3.3.0
cleverhans>=4.0.0

# Differential privacy
opacus>=1.4.0  # PyTorch
tensorflow-privacy>=0.9.0  # TensorFlow
```

## requirements/rag.txt
RAG and vector database dependencies:

```txt
-r base.txt
-r transformers.txt

# LLM frameworks
langchain>=0.0.350
llama-index>=0.9.0

# Vector databases
chromadb>=0.4.0
faiss-cpu>=1.7.4  # or faiss-gpu
qdrant-client>=1.7.0
pinecone-client>=2.2.0

# Document processing
pypdf>=3.17.0
python-docx>=1.1.0
markdown>=3.5.0
beautifulsoup4>=4.12.0
```

---

# Installation Guide

## Quick Start (Basic Installation)

```bash
# Clone repository
git clone https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey.git
cd PearlMind-ML-Journey

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with base dependencies
pip install -e .
```

## Framework-Specific Installation

### PyTorch Setup
```bash
# CPU version
pip install -e ".[pytorch]"

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[pytorch]"
```

### TensorFlow Setup
```bash
pip install -e ".[tensorflow]"
```

### Full Development Environment
```bash
# Install everything for development
pip install -e ".[all,dev]"

# Set up pre-commit hooks
pre-commit install
```

## Docker Installation

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/

# Install package
RUN pip install --no-cache-dir -e ".[production]"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PEARLMIND_ENV=production

# Run the application
CMD ["pearlmind-serve"]
```

## Environment Variables

Create a `.env` file:

```env
# Environment
PEARLMIND_ENV=development

# Paths
PEARLMIND_DATA_PATH=./data
PEARLMIND_MODEL_PATH=./models
PEARLMIND_LOG_PATH=./logs

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ML Configuration
DEFAULT_DEVICE=cpu  # or cuda
DEFAULT_BATCH_SIZE=32
DEFAULT_SEED=42

# Monitoring
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your-key-here

# Security
ENABLE_FAIRNESS_AUDIT=true
ENABLE_ADVERSARIAL_TESTING=true
```

## Makefile for Common Tasks

Create a `Makefile`:

```makefile
.PHONY: help install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  install      Install package with base dependencies"
	@echo "  install-dev  Install package with all dev dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=pearlmind --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

## GitHub Actions Setup

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        black --check src/ tests/
        isort --check-only src/ tests/
        flake8 src/ tests/
    
    - name: Run tests
      run: |
        pytest tests/ --cov=pearlmind --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```
