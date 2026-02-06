.PHONY: help build up down restart logs test clean install lint format

# Default target
help:
	@echo "AdInsights-Agent - Makefile commands"
	@echo ""
	@echo "  make build     - Build Docker image"
	@echo "  make up        - Start services"
	@echo "  make down      - Stop services"
	@echo "  make restart   - Restart services"
	@echo "  make logs      - View logs"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up containers and volumes"
	@echo "  make install   - Install dependencies locally"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"

# Build Docker image
build:
	docker-compose build

# Start services
up:
	docker-compose up -d
	@echo "API available at http://localhost:8000"
	@echo "Docs available at http://localhost:8000/docs"

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f adinsights-api

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Clean up
clean:
	docker-compose down -v
	rm -rf htmlcov/ .pytest_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Install dependencies
install:
	pip install -r requirements.txt

# Lint code
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Run with cache profile
up-with-cache:
	docker-compose --profile cache up -d

# Development mode with hot reload
dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
