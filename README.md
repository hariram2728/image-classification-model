# image-classification-model

Professional image classification system with separate backend (FastAPI) and frontend (Streamlit).

## 📁 Project Structure

```
your-repo/
├── api/                  # FastAPI backend
│   ├── main.py          # API endpoints
│   └── requirements.txt # Backend dependencies
├── frontend/            # Streamlit frontend
│   ├── app.py          # UI application
│   └── requirements.txt # Frontend dependencies
├── models/              # Model files
│   ├── model.pth       # Trained model weights
│   └── class_labels.json # Class mappings
└── README.md           # This file
```

## 🚀 Quick Start

### Local Development

**1. Setup Backend:**
```bash
cd api
pip install -r requirements.txt
python main.py
```
Backend runs on http://localhost:8000

**2. Setup Frontend (in new terminal):**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Frontend runs on http://localhost:8501

## ☁️ Cloud Deployment

### Deploy Backend (Render.com)

1. Create account on [Render](https://render.com)
2. Create new **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Root Directory:** `api`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:**
     - `MODEL_PATH`: `models/model.pth`
     - `LABELS_PATH`: `models/class_labels.json`
5. Deploy! You'll get a URL like: `https://your-api.onrender.com`

### Deploy Frontend (Streamlit Cloud)

1. Create account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **New App**
3. Connect your GitHub repository
4. Configure:
   - **Main File Path:** `frontend/app.py`
   - **Working Directory:** (leave blank)
   - **Python Version:** 3.11
5. Add Secrets (click "Secrets" button):
   ```toml
   BACKEND_URL = "https://your-api.onrender.com"
   ```
6. Deploy! You'll get a URL like: `https://your-app.streamlit.app`

## 🔧 Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `MODEL_PATH` | models/model.pth | Path to model weights |
| `LABELS_PATH` | models/class_labels.json | Path to class labels |

### Frontend Secrets

| Secret | Description |
|--------|-------------|
| `BACKEND_URL` | URL of deployed backend API |

## 📝 Usage

1. Open the Streamlit app URL
2. Upload an image (JPG, PNG, WebP, BMP)
3. View prediction results with confidence scores
4. Check sidebar for backend connection status

## 🧪 Testing

**Test Backend:**
```bash
curl http://localhost:8000/health
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

**Test Frontend:**
Open http://localhost:8501 in browser

## 📊 API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Predict image class
- `GET /docs` - Interactive API documentation

## ⚠️ Important Notes

1. **Model File**: You need to train and save your model as `models/model.pth`
2. **Class Labels**: Update `models/class_labels.json` with your actual classes
3. **CORS**: Update CORS settings in `api/main.py` for production
4. **Free Tier Limits**: 
   - Render: Free tier sleeps after 15 min inactivity
   - Streamlit Cloud: Free for public repos

## 💰 Cost

- **Backend (Render)**: Free tier available
- **Frontend (Streamlit)**: Free for public apps
- **Total**: $0/month for development

## 🛠️ Troubleshooting

**Backend not starting?**
- Check logs on Render dashboard
- Verify model file exists in repository
- Ensure PORT environment variable is set

**Frontend can't connect?**
- Verify BACKEND_URL in Streamlit secrets
- Check if backend is running (visit /health endpoint)
- Ensure CORS allows your frontend URL

**Model loading errors?**
- Verify model architecture matches code
- Check file paths are correct
- Ensure model was saved properly

## 📄 License

MIT License - Feel free to use for client projects!
