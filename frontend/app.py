"""
Streamlit Frontend for Image Classification
Deploy this folder separately on Streamlit Cloud
"""
import streamlit as st
import requests
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
    }
    .confidence-fill {
        background-color: #4CAF50;
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #999;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Get backend URL from secrets or environment
def get_backend_url():
    """Get backend API URL from Streamlit secrets or environment"""
    try:
        # Try Streamlit secrets first
        return st.secrets["BACKEND_URL"]
    except:
        # Fallback to environment variable
        return os.getenv("BACKEND_URL", "http://localhost:8000")

# Header
st.markdown('<div class="main-header">🔍 Image Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image to classify it using our AI model</div>', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This application uses deep learning to classify images into different categories.")
    st.write("**Supported formats:** JPG, PNG, WebP, BMP")
    st.write("**Max file size:** 10 MB")
    
    st.divider()
    
    st.header("📊 Model Info")
    backend_url = get_backend_url()
    st.write(f"Backend: `{backend_url}`")
    
    # Try to get health status
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
            health_data = response.json()
            st.write(f"- Status: {health_data.get('status', 'unknown')}")
            st.write(f"- Device: {health_data.get('device', 'unknown')}")
        else:
            st.error("❌ Backend Error")
    except:
        st.warning("⚠️ Cannot connect to backend")

# Main content
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Select an image file to classify"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        try:
            # Reset pointer just in case
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            # Convert to RGB to handle potential RGBA/P issues
            if image.mode != "RGB":
                image = image.convert("RGB")
            st.image(image, use_column_width=True)  # Older parameter name
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
        
        # Image details
        st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"**Format:** {image.format}")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Show loading spinner while predicting
        with st.spinner("Analyzing image..."):
            try:
                # Prepare file for upload
                img_bytes = io.BytesIO()
                image.save(img_bytes, format=image.format or 'JPEG')
                img_bytes = img_bytes.getvalue()
                
                # Send to backend
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                response = requests.post(
                    f"{backend_url}/predict",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="margin: 0;">Prediction: {result['prediction']}</h3>
                        <p style="margin: 10px 0; color: #666;">Confidence Score</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {result['confidence']}%;"></div>
                        </div>
                        <p style="text-align: right; font-weight: bold; color: #4CAF50;">{result['confidence']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all probabilities
                    st.write("**All Class Probabilities:**")
                    sorted_probs = sorted(
                        result.get('all_probabilities', {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for label, prob in sorted_probs:
                        st.write(f"- **{label}**: {prob}%")
                    
                    st.success("✅ Classification successful!")
                    
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend server. Please check if the API is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload an image to get started")
    
    # Example section
    st.divider()
    st.subheader("📝 How to use")
    st.write("""
    1. Click the **'Browse files'** button above
    2. Select an image from your computer
    3. Wait for the AI to analyze the image
    4. View the prediction and confidence score
    """)

# Footer
st.markdown('<div class="footer">Powered by Deep Learning • Built with Streamlit & FastAPI</div>', unsafe_allow_html=True)

# Hide Streamlit branding
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
