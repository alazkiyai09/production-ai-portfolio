.PHONY: help install test lint format clean run-api dev

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

test: ## Run tests
	. .venv/bin/activate && pytest -v

test-cov: ## Run tests with coverage
	. .venv/bin/activate && pytest --cov=src --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	. .venv/bin/activate && pytest -m unit -v

test-integration: ## Run integration tests only
	. .venv/bin/activate && pytest -m integration -v

test-no-llm: ## Run tests without LLM calls (faster)
	. .venv/bin/activate && pytest -m "not llm" -v

lint: ## Run linting
	. .venv/bin/activate && ruff check src/ tests/

format: ## Format code with black and ruff
	. .venv/bin/activate && black src/ tests/
	. .venv/bin/activate && ruff check --fix src/ tests/

type-check: ## Run type checking with mypy
	. .venv/bin/activate && mypy src/

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage
	rm -rf dist/ build/ *.egg-info

run-api: ## Run the API server
	. .venv/bin/activate && python -m src.api.main

dev: ## Run API with auto-reload
	. .venv/bin/activate && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

check: lint type-check test ## Run all checks (lint, type-check, test)

all: install check ## Install and run all checks
