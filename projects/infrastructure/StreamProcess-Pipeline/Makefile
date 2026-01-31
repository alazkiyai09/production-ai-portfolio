.PHONY: help install dev test lint format clean build docker-build docker-up docker-down k8s-deploy k8s-delete

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

## help: Show this help message
help:
	@echo '$(BLUE)StreamProcess-Pipeline Makefile$(NC)'
	@echo ''
	@echo 'Usage:'
	@echo '  make $(GREEN)<target>$(NC)'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## install: Install dependencies
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "$(GREEN)Dependencies installed!$(NC)"

## dev: Install development dependencies
dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e ".[dev,kafka,qdrant]"
	@echo "$(GREEN)Development dependencies installed!$(NC)"

## test: Run tests
test:
	@echo "$(BLUE)Running tests...$(NC)"
	pytest -v
	@echo "$(GREEN)Tests completed!$(NC)"

## test-cov: Run tests with coverage
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

## test-unit: Run unit tests only
test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit -v

## test-integration: Run integration tests only
test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration -v

## lint: Run linters
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	ruff check src/ tests/
	@echo "$(GREEN)Linting completed!$(NC)"

## lint-fix: Auto-fix linting issues
lint-fix:
	@echo "$(BLUE)Auto-fixing linting issues...$(NC)"
	ruff check --fix src/ tests/

## format: Format code with Black
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

## format-check: Check code formatting
format-check:
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src/ tests/

## type-check: Run type checker
type-check:
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy src/
	@echo "$(GREEN)Type checking completed!$(NC)"

## check-all: Run all checks (lint, format, type-check)
check-all: lint format-check type-check
	@echo "$(GREEN)All checks passed!$(NC)"

## clean: Clean up generated files
clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "$(GREEN)Cleanup completed!$(NC)"

## build: Build the package
build:
	@echo "$(BLUE)Building package...$(NC)"
	python -m build
	@echo "$(GREEN)Package built!$(NC)"

## docker-build: Build Docker image
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)Docker image built!$(NC)"

## docker-up: Start Docker services
docker-up:
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Docker services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Flower: http://localhost:5555"
	@echo "Prometheus: http://localhost:9091"
	@echo "Grafana: http://localhost:3000"

## docker-down: Stop Docker services
docker-down:
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Docker services stopped!$(NC)"

## docker-logs: Show Docker logs
docker-logs:
	docker-compose logs -f

## docker-restart: Restart Docker services
docker-restart:
	@echo "$(BLUE)Restarting Docker services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)Docker services restarted!$(NC)"

## k8s-deploy: Deploy to Kubernetes
k8s-deploy:
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/postgres-statefulset.yaml
	kubectl apply -f k8s/redis-deployment.yaml
	kubectl apply -f k8s/chroma-deployment.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/worker-deployment.yaml
	kubectl apply -f k8s/flower-deployment.yaml
	kubectl apply -f k8s/network-policy.yaml
	kubectl apply -f k8s/ingress.yaml
	@echo "$(GREEN)Deployed to Kubernetes!$(NC)"
	@echo "Waiting for deployments..."
	kubectl wait --for=condition=available --timeout=300s \
		deployment/streamprocess-api \
		deployment/streamprocess-worker \
		deployment/streamprocess-flower \
		-n streamprocess

## k8s-delete: Delete Kubernetes resources
k8s-delete:
	@echo "$(BLUE)Deleting Kubernetes resources...$(NC)"
	kubectl delete -f k8s/ingress.yaml --ignore-not-found=true
	kubectl delete -f k8s/network-policy.yaml --ignore-not-found=true
	kubectl delete -f k8s/flower-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/worker-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/api-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/chroma-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/redis-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/postgres-statefulset.yaml --ignore-not-found=true
	kubectl delete -f k8s/secret.yaml --ignore-not-found=true
	kubectl delete -f k8s/configmap.yaml --ignore-not-found=true
	kubectl delete -f k8s/namespace.yaml --ignore-not-found=true
	@echo "$(GREEN)Kubernetes resources deleted!$(NC)"

## k8s-status: Show Kubernetes status
k8s-status:
	@echo "$(BLUE)Kubernetes Resources Status:$(NC)"
	@echo ""
	kubectl get all -n streamprocess

## k8s-logs-api: Show API logs
k8s-logs-api:
	kubectl logs -f -l app=streamprocess-api -n streamprocess

## k8s-logs-worker: Show worker logs
k8s-logs-worker:
	kubectl logs -f -l app=streamprocess-worker -n streamprocess

## k8s-port-forward: Port forward to API
k8s-port-forward:
	kubectl port-forward deployment/streamprocess-api 8000:8000 -n streamprocess

## migrate: Run database migrations
migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)Migrations completed!$(NC)"

## migration-create: Create new migration (usage: make migration-create MSG="add_user_table")
migration-create:
	@echo "$(BLUE)Creating migration: $(MSG)$(NC)"
	alembic revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)Migration created!$(NC)"

## worker: Run Celery worker locally
worker:
	@echo "$(BLUE)Starting Celery worker...$(NC)"
	celery -A src.processing.worker worker --loglevel=info --concurrency=4

## flower: Run Flower locally
flower:
	@echo "$(BLUE)Starting Flower...$(NC)"
	celery -A src.processing.worker flower --port=5555

## api: Run API server locally
api:
	@echo "$(BLUE)Starting API server...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

## shell: Open Python shell
shell:
	@echo "$(BLUE)Opening Python shell...$(NC)"
	python
