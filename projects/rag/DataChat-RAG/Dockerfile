# Multi-stage Dockerfile for DataChat-RAG
# Stage 1: Base image with dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Development
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-mock \
    black \
    ruff \
    mypy \
    ipython \
    ipdb

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data/chromadb /app/data/uploads /app/logs

# Set permissions
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# Expose ports
EXPOSE 8000 8501

# Default command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production
FROM dependencies as production

# Install production-only dependencies
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create directories with proper permissions
RUN mkdir -p /app/data/chromadb /app/data/uploads /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Switch to non-root user
USER appuser
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Run with gunicorn in production
CMD ["gunicorn", "src.api.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info"]

# Stage 5: Streamlit UI
FROM base as streamlit

# Copy requirements
COPY requirements.txt .

# Install Streamlit and dependencies
RUN pip install --no-cache-dir streamlit requests pandas

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/

# Create directories
RUN mkdir -p /app/data/chromadb /app/data/uploads

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/ui/chat_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
