#!/bin/bash
# Quick Deployment Script for Image Classification API
# Minimal dependencies, works on low-resource machines

set -e

echo "🚀 Image Classification API - Quick Setup"
echo "=========================================="

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install minimal dependencies
echo "📥 Installing minimal dependencies (this may take 2-3 minutes)..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch==2.0.1 \
    torchvision==0.15.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    pillow==10.1.0 \
    pydantic==2.5.0

echo "✅ Dependencies installed!"

# Create artifacts directory
mkdir -p artifacts/models

# Check if model exists
if [ ! -f "artifacts/models/model.pth" ]; then
    echo "⚠️  No model found at artifacts/models/model.pth"
    echo ""
    echo "To export your trained model, run:"
    echo "  python artifacts/scripts/export_model.py --checkpoint YOUR_CHECKPOINT.pth --output-dir artifacts/models"
    echo ""
    echo "Or download a pretrained model manually."
    echo ""
    echo "You can still start the API (it will return 'model not loaded' errors)."
fi

# Show usage instructions
echo ""
echo "=========================================="
echo "✨ Setup Complete!"
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  python artifacts/scripts/inference_api.py"
echo ""
echo "Then visit: http://localhost:8000/docs"
echo ""
echo "API Endpoints:"
echo "  GET  /health          - Health check"
echo "  POST /predict         - Upload image file"
echo "  POST /predict-base64  - Send base64 image"
echo ""
echo "Example prediction:"
echo "  curl -X POST http://localhost:8000/predict \\"
echo "       -F 'file=@your_image.jpg' \\"
echo "       -F 'top_k=5'"
echo ""
echo "=========================================="
echo ""

# Ask if user wants to start the server
read -p "Do you want to start the API server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting API server..."
    python artifacts/scripts/inference_api.py
fi
