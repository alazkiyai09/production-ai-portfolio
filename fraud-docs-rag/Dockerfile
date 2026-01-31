# Multi-stage Dockerfile for FraudDocs-RAG
#
# Stage 1: Builder - Compile dependencies and create virtual environment
# Stage 2: Runtime - Minimal image with just the application
#
# Usage:
#   docker build -t frauddocs-rag:latest .
#   docker run -p 8000:8000 --env-file .env frauddocs-rag:latest

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1

# Set environment variables for build
ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download models during build (optional - will cache models)
# This reduces startup time but increases image size
# RUN python -c "from sentence_transformers import SentenceTransformer; \
#                SentenceTransformer('BAAI/bge-small-en-v1.5')"


# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim AS runtime

# Set runtime arguments
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

# Set environment variables
ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE} \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    ENVIRONMENT="production" \
    HOST="0.0.0.0" \
    PORT="8000"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    curl \
    # Libraries for PDF/document processing
    libffi8 \
    libssl3 \
    # Libraries for machine learning models
    libgomp1 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app/data/documents \
             /app/data/raw \
             /app/data/chroma_db \
             /app/data/vector_store \
             /app/logs \
    && chown -R ${APP_USER}:${APP_USER} /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} src/ /app/src/
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml /app/

# Create necessary __pycache__ directories with correct permissions
RUN python -c "import sys; sys.path.insert(0, '/app/src')" \
    || true

# Switch to non-root user
USER ${APP_USER}

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "fraud_docs_rag.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]


# ============================================================================
# Development Stage (Optional)
# ============================================================================
FROM runtime AS development

# Switch back to root for development tools
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt \
    && pip install pytest pytest-cov black ruff mypy

# Switch back to app user
USER ${APP_USER}

# Override command for development with hot reload
CMD ["uvicorn", "fraud_docs_rag.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]


# ============================================================================
# Build Arguments Documentation
# ============================================================================
#
# To build specific stages:
#
# Production (minimal):
#   docker build --target runtime -t frauddocs-rag:latest .
#
# Development (with dev tools):
#   docker build --target development -t frauddocs-rag:dev .
#
# With custom arguments:
#   docker build \
#     --build-arg APP_USER=myuser \
#     --build-arg APP_UID=1001 \
#     -t frauddocs-rag:latest .
#
# View build details:
#   docker history frauddocs-rag:latest
#
