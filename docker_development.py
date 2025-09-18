# Dockerfile
# Multi-stage build for PearlMind ML Journey

# Stage 1: Base image with system dependencies
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash pearlmind && \
    mkdir -p /app /home/pearlmind/.cache && \
    chown -R pearlmind:pearlmind /app /home/pearlmind

# Set working directory
WORKDIR /app

# ============================================
# Stage 2: Development environment
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements/ requirements/
COPY pyproject.toml .

# Install all dependencies including dev tools
RUN pip install --upgrade pip && \
    pip install -e ".[dev,fairness,security]"

# Switch to non-root user
USER pearlmind

# Copy source code
COPY --chown=pearlmind:pearlmind . .

# Expose ports for Jupyter and API
EXPOSE 8888 8000

# Default command for development
CMD ["bash"]

# ============================================
# Stage 3: Production environment
FROM base AS production

# Copy only necessary files
COPY requirements/base.txt requirements/production.txt requirements/
COPY pyproject.toml .

# Install production dependencies only
RUN pip install --upgrade pip && \
    pip install -r requirements/production.txt

# Switch to non-root user
USER pearlmind

# Copy source code and configs
COPY --chown=pearlmind:pearlmind src/ src/
COPY --chown=pearlmind:pearlmind configs/ configs/

# Install package
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the production server
CMD ["pearlmind", "serve", "models/production/model.pkl"]

# ============================================
# Stage 4: Jupyter notebook environment
FROM development AS jupyter

# Expose Jupyter port
EXPOSE 8888

# Set Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Default command for Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

---

# docker-compose.yml
version: '3.8'

services:
  # Development environment
  dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    image: pearlmind:dev
    container_name: pearlmind-dev
    volumes:
      - .:/app
      - pearlmind-cache:/home/pearlmind/.cache
    environment:
      - PEARLMIND_ENV=development
      - PYTHONPATH=/app
    ports:
      - "8000:8000"  # API
      - "8888:8888"  # Jupyter
      - "5000:5000"  # MLflow
    command: bash
    stdin_open: true
    tty: true

  # Jupyter notebook server
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
      target: jupyter
    image: pearlmind:jupyter
    container_name: pearlmind-jupyter
    volumes:
      - .:/app
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
    ports:
      - "8888:8888"
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  # API server
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: pearlmind:api
    container_name: pearlmind-api
    volumes:
      - ./models:/app/models:ro
      - ./configs:/app/configs:ro
    environment:
      - PEARLMIND_ENV=production
      - API_HOST=0.0.0.0
      - API_PORT=8000
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # MLflow tracking server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: pearlmind-mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
      - ./mlruns:/mlruns
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlruns

  # PostgreSQL for production metadata
  postgres:
    image: postgres:15-alpine
    container_name: pearlmind-postgres
    environment:
      - POSTGRES_DB=pearlmind
      - POSTGRES_USER=pearlmind
      - POSTGRES_PASSWORD=pearlmind123
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pearlmind"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: pearlmind-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pearlmind-cache:
  mlflow-data:
  postgres-data:
  redis-data:

networks:
  default:
    name: pearlmind-network

---

# .dockerignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# ML artifacts
*.h5
*.pkl
*.pth
*.onnx
mlruns/
outputs/
logs/

# Data (except configs)
data/raw/*
data/processed/*
!data/.gitkeep

# Models (except production)
models/experiments/*
models/checkpoints/*
!models/production/.gitkeep

# OS
.DS_Store
Thumbs.db

# Git
.git
.gitignore

---

# Makefile for Docker commands
.PHONY: docker-build docker-up docker-down docker-clean docker-dev docker-jupyter docker-api

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting all services..."
	docker-compose up -d

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

docker-dev:
	@echo "Starting development environment..."
	docker-compose up dev

docker-jupyter:
	@echo "Starting Jupyter server..."
	docker-compose up -d jupyter
	@echo "Jupyter available at http://localhost:8888"

docker-api:
	@echo "Starting API server..."
	docker-compose up -d api
	@echo "API available at http://localhost:8000"

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec dev bash

docker-test:
	docker-compose exec dev pytest tests/ -v

docker-lint:
	docker-compose exec dev make lint

---

# scripts/docker-entrypoint.sh
#!/bin/bash
set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Wait for postgres to be ready (if needed)
if [ "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    while ! nc -z postgres 5432; do
        sleep 0.1
    done
    echo "PostgreSQL started"
fi

# Run migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # Add migration commands here
fi

# Execute the main command
exec "$@"

---

# Development environment setup script
# scripts/docker-setup.sh
#!/bin/bash

echo "🚀 Setting up PearlMind Docker environment..."

# Create necessary directories
mkdir -p data/{raw,processed,features,cache}
mkdir -p models/{baseline,experiments,production,registry}
mkdir -p logs/{training,evaluation,serving}
mkdir -p mlruns
mkdir -p notebooks

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
# PearlMind Environment Configuration
PEARLMIND_ENV=development

# Database
DATABASE_URL=postgresql://pearlmind:pearlmind123@postgres:5432/pearlmind

# Redis
REDIS_URL=redis://redis:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
DEFAULT_DEVICE=cpu
DEFAULT_BATCH_SIZE=32

# Security
SECRET_KEY=your-secret-key-here
EOF
    echo "✓ Created .env file"
fi

# Build Docker images
docker-compose build

echo "✓ Docker environment ready!"
echo ""
echo "Available commands:"
echo "  make docker-up       - Start all services"
echo "  make docker-dev      - Start development environment"
echo "  make docker-jupyter  - Start Jupyter server"
echo "  make docker-api      - Start API server"
echo "  make docker-shell    - Open shell in dev container"
echo "  make docker-test     - Run tests in container"
echo ""
echo "Services:"
echo "  Development: http://localhost:8000"
echo "  Jupyter:     http://localhost:8888"
echo "  MLflow:      http://localhost:5000"
echo "  PostgreSQL:  localhost:5432"
echo "  Redis:       localhost:6379"
