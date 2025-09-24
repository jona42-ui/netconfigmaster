# Multi-stage Dockerfile for NetConfigMaster
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Development stage
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry install --with dev,docs && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Install pre-commit hooks
RUN poetry run pre-commit install || true

# Set the default command
CMD ["poetry", "shell"]

# Production stage
FROM base as production

# Install only production dependencies
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port for web UI
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health_check || exit 1

# Default command
CMD ["poetry", "run", "python", "src/ui.py"]

# Training stage (for model training workloads)
FROM base as training

# Install dependencies including CUDA support
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Install additional training dependencies
RUN apt-get update && apt-get install -y \
    nvidia-ml-py3 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Set up for training
WORKDIR /app

CMD ["poetry", "run", "python", "src/train.py"]

# Evaluation stage
FROM base as evaluation

# Install dependencies
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

CMD ["poetry", "run", "python", "src/model_evaluation.py"]
