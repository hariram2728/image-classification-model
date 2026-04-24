from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-classification",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready end-to-end image classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-classification",
    packages=find_packages(where=".", include=["src*", "api*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "training": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "optuna>=3.3.0",
        ],
        "api": [
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
        ],
        "explainability": [
            "grad-cam>=1.4.0",
            "lime>=0.2.0",
            "shap>=0.42.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train=src.training.trainer:main",
            "evaluate=src.evaluation.evaluator:main",
            "predict=src.inference.predictor:main",
            "api=api.main:main",
        ],
    },
)
