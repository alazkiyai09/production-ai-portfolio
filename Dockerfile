# Multi-stage Dockerfile for AgenticFlow
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set build arguments
ARG POETRY_VERSION=1.7.1
ARG PYTHON_VERSION=3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install build tools
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set labels
LABEL maintainer="AgenticFlow Team"
LABEL description="Multi-Agent Workflow System with LangGraph"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    AGENTICFLOW_HOME=/app \
    WORKSPACE_ROOT=/app/workspace

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r agenticflow && \
    useradd -r -g agenticflow -d /app -s /bin/bash \
    -c "AgenticFlow user" agenticflow

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory structure
WORKDIR /app
RUN mkdir -p /app/workspace /app/data /app/logs && \
    chown -R agenticflow:agenticflow /app

# Copy application files
COPY --chown=agenticflow:agenticflow src/ /app/src/
COPY --chown=agenticflow:agenticflow pyproject.toml /app/
COPY --chown=agenticflow:agenticflow requirements.txt /app/

# Switch to non-root user
USER agenticflow

# Expose ports
# 8000: FastAPI backend
# 8501: Streamlit UI
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: UI-only variant
FROM runtime as ui

# Override command for UI
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address= "0.0.0.0"]
