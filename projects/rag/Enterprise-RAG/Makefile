# ============================================================
# Enterprise-RAG: Makefile for Common Tasks
# ============================================================
# This Makefile provides convenient commands for development

.PHONY: help install dev test lint format clean run-api run-ui docker-build docker-up

# Default target
.DEFAULT_GOAL := help

# ============================================================
# Development Setup
# ============================================================
install: ## Install dependencies
	@echo "Installing dependencies..."
	pip install -e .
	pip install -e ".[dev]"

dev: ## Install development dependencies
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pre-commit install

# ============================================================
# Code Quality
# ============================================================
lint: ## Run linting checks
	@echo "Running linting..."
	ruff check src/
	ruff check tests/
	mypy src/

format: ## Format code with black and ruff
	@echo "Formatting code..."
	black src/ tests/
	ruff check --fix src/ tests/

format-check: ## Check if code is formatted
	@echo "Checking code format..."
	black --check src/ tests/
	ruff check src/ tests/

# ============================================================
# Testing
# ============================================================
test: ## Run all tests
	@echo "Running tests..."
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-fast: ## Run fast tests (skip slow)
	@echo "Running fast tests..."
	pytest tests/ -v -m "not slow"

# ============================================================
# Running the Application
# ============================================================
run-api: ## Run the FastAPI server
	@echo "Starting FastAPI server..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui: ## Run the Streamlit UI
	@echo "Starting Streamlit UI..."
	streamlit run src/ui/app.py

run-dev: ## Run both API and UI (in background)
	@echo "Starting API and UI..."
	@uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 & \
	streamlit run src/ui/app.py

# ============================================================
# Docker
# ============================================================
docker-build: ## Build Docker images
	@echo "Building Docker images..."
	docker-compose build

docker-up: ## Start Docker containers
	@echo "Starting Docker containers..."
	docker-compose up -d

docker-down: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-clean: ## Remove Docker containers and volumes
	@echo "Cleaning Docker..."
	docker-compose down -v
	docker system prune -f

# ============================================================
# Utilities
# ============================================================
clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov/ *.egg-info/

setup-dirs: ## Create necessary directories
	@echo "Creating directories..."
	mkdir -p data/documents data/chroma logs

download-models: ## Download embedding and reranking models
	@echo "Downloading models..."
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
	python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ============================================================
# Help
# ============================================================
help: ## Show this help message
	@echo "Enterprise-RAG Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
