.PHONY: help install dev train evaluate test lint format clean docker-build docker-run api

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

train: ## Run model training
	python -m src.training.trainer --config configs/training_config.yaml

evaluate: ## Run model evaluation
	python -m src.evaluation.evaluator --config configs/training_config.yaml

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-performance: ## Run performance tests only
	pytest tests/performance/ -v

lint: ## Run linters
	flake8 src/ api/
	black --check src/ api/
	isort --check-only src/ api/
	mypy src/ api/

format: ## Format code
	black src/ api/
	isort src/ api/

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -rf coverage.xml .coverage

docker-build-api: ## Build API Docker image
	docker build -f infrastructure/docker/Dockerfile.api -t image-classification-api:latest .

docker-build-training: ## Build training Docker image
	docker build -f infrastructure/docker/Dockerfile.training -t image-classification-training:latest .

docker-build-inference: ## Build inference Docker image
	docker build -f infrastructure/docker/Dockerfile.inference -t image-classification-inference:latest .

docker-run-api: ## Run API container
	docker run -p 8000:8000 --env-file .env image-classification-api:latest

api: ## Start the FastAPI server locally
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

preprocess: ## Preprocess data
	python -m src.data.preprocessing --config configs/data_config.yaml

augment: ## Augment training data
	python -m src.data.augmentation --config configs/data_config.yaml

export-model: ## Export model to ONNX/TorchScript
	python scripts/export_model.sh

mlflow-server: ## Start MLflow tracking server
	mlflow server --host 0.0.0.0 --port 5000

download-data: ## Download dataset
	bash scripts/download_data.sh

setup-env: ## Setup development environment
	bash scripts/setup_environment.sh

deploy-staging: ## Deploy to staging environment
	bash scripts/deploy.sh staging

deploy-production: ## Deploy to production environment
	bash scripts/deploy.sh production

helm-install: ## Install Helm chart
	helm install image-classifier infrastructure/helm/image-classifier -f infrastructure/helm/image-classifier/values.yaml

helm-upgrade: ## Upgrade Helm release
	helm upgrade image-classifier infrastructure/helm/image-classifier -f infrastructure/helm/image-classifier/values-production.yaml

terraform-init: ## Initialize Terraform
	cd infrastructure/terraform && terraform init

terraform-plan: ## Plan Terraform changes
	cd infrastructure/terraform && terraform plan

terraform-apply: ## Apply Terraform changes
	cd infrastructure/terraform && terraform apply -auto-approve

security-scan: ## Run security scans
	bandit -r src/ api/ -ll
	safety check -r requirements.txt

notebook: ## Start Jupyter notebook
	jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8888

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve
