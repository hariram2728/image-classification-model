# 🚀 Quick Deployment Guide

## No Docker Required - Minimal Setup

This guide shows how to deploy your image classification model with **minimal disk space** (~2GB total).

## Prerequisites

- Python 3.8+
- 2GB free disk space
- Internet connection for initial download

## Step 1: Export Your Model

First, export your trained model to deployment format:

```bash
python artifacts/scripts/export_model.py \
    --checkpoint path/to/your/checkpoint.pth \
    --output-dir artifacts/models \
    --model-name resnet50 \
    --num-classes 10
```

This creates:
- `artifacts/models/model.pth` - PyTorch model
- `artifacts/models/class_labels.json` - Class names
- `artifacts/models/model.onnx` - ONNX format (optional)
- `artifacts/models/model.pt` - TorchScript (optional)

## Step 2: Quick Deploy Script

Run the automated setup script:

```bash
bash artifacts/scripts/quick_deploy.sh
```

This will:
1. Create a virtual environment (~50MB)
2. Install minimal dependencies (~1.5GB for CPU PyTorch)
3. Start the API server

**Total disk usage: ~2GB**

## Step 3: Use the API

Once running, visit: **http://localhost:8000/docs**

### Test with curl:

```bash
# Health check
curl http://localhost:8000/health

# Predict from file
curl -X POST http://localhost:8000/predict \
     -F 'file=@test_image.jpg' \
     -F 'top_k=5'

# Predict from base64
curl -X POST http://localhost:8000/predict-base64 \
     -H "Content-Type: application/json" \
     -d '{
           "image_base64": "YOUR_BASE64_STRING",
           "top_k": 5
         }'
```

## Manual Installation (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies
pip install torch torchvision fastapi uvicorn python-multipart pillow pydantic

# Start API
python artifacts/scripts/inference_api.py
```

## Deploy to Free Cloud

### Option 1: Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Choose "Docker" as SDK
3. Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install torch torchvision fastapi uvicorn python-multipart pillow

CMD ["python", "artifacts/scripts/inference_api.py"]
```

4. Push your code to the Space
5. It deploys automatically!

### Option 2: Google Colab

```python
# In a Colab notebook:
!pip install fastapi uvicorn python-multipart pillow torch torchvision

# Upload your files
from google.colab import files
files.upload()  # Upload inference_api.py and model.pth

# Run in background
!nohup python inference_api.py &

# Expose with ngrok
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
```

### Option 3: Render.com

1. Create new Web Service on https://render.com
2. Connect your GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python artifacts/scripts/inference_api.py`
5. Deploy!

## Requirements File

Create `requirements.txt` for cloud deployment:

```txt
torch==2.0.1
torchvision==0.15.2
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pillow==10.1.0
pydantic==2.5.0
```

## Performance Tips

- Use CPU-only PyTorch to save 2GB (no CUDA)
- Quantize model: `torch.quantization.quantize_dynamic()`
- Use smaller models: MobileNetV2, EfficientNet-B0
- Enable model caching in production

## Troubleshooting

**Model not loading?**
- Check path: `ls artifacts/models/`
- Verify checkpoint format
- Check logs for errors

**Out of memory?**
- Reduce batch size
- Use smaller model architecture
- Enable gradient checkpointing

**Slow inference?**
- Use ONNX Runtime: `pip install onnxruntime`
- Enable quantization
- Resize input images

## Next Steps

- Add authentication
- Set up monitoring
- Configure auto-scaling
- Add logging to file
- Set up CI/CD pipeline

---

**Need help?** Check the full documentation in `/docs` folder.
