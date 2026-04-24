# Image Classification Project

A production-ready, end-to-end image classification system with MLOps best practices.

## Features

- **Data Pipeline**: Automated data loading, validation, preprocessing, and augmentation
- **Model Training**: Support for multiple architectures (ResNet, EfficientNet, ViT, Custom CNN)
- **Evaluation**: Comprehensive metrics, confusion matrix, and benchmarking
- **Explainability**: GradCAM, LIME, SHAP, and attention visualization
- **API Service**: FastAPI-based inference service with authentication and rate limiting
- **Monitoring**: Data drift, concept drift, and model performance monitoring
- **CI/CD**: GitHub Actions workflows for testing, training, and deployment
- **Infrastructure**: Docker, Kubernetes, Terraform, and Helm configurations

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd image-classification

# Setup environment
./scripts/setup_environment.sh

# Download and preprocess data
./scripts/download_data.sh
./scripts/preprocess_data.sh

# Train model
./scripts/train_model.sh

# Run tests
./scripts/run_tests.sh

# Start API server
docker-compose up
```

## Project Structure

See the detailed project structure in the repository. Key directories:

- `src/`: Core source code
- `api/`: FastAPI application
- `pipelines/`: ML pipelines
- `infrastructure/`: Deployment configurations
- `tests/`: Test suites
- `configs/`: Configuration files

## Documentation

- [Architecture](docs/architecture.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Model Card](docs/model_card.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Monitoring Guide](docs/monitoring_guide.md)

## License

MIT License
