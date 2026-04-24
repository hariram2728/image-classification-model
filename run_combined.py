#!/usr/bin/env python3
"""
Unified Launcher for Production Deployment
Starts both the FastAPI Backend and Streamlit Frontend in a single process.
Perfect for single-container deployments (Render, Railway, Hugging Face Spaces).
"""
import subprocess
import sys
import os
import time
import signal
import threading

# Configuration
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", 8000))
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
STREAMLIT_SERVER_ADDRESS = "0.0.0.0"

# Paths
API_SCRIPT = "inference_api.py"
FRONTEND_SCRIPT = "frontend_app.py"

def run_api():
    """Run the FastAPI backend using uvicorn"""
    print(f"🚀 Starting FastAPI Backend on http://{API_HOST}:{API_PORT}")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "inference_api:app",  # Assumes app is defined in inference_api.py
        "--host", API_HOST,
        "--port", str(API_PORT),
        "--workers", "1",
        "--log-level", "info"
    ]
    
    # Use subprocess to run uvicorn
    process = subprocess.Popen(cmd)
    return process

def run_frontend():
    """Run the Streamlit frontend"""
    print(f"🎨 Starting Streamlit Frontend on http://{STREAMLIT_SERVER_ADDRESS}:{STREAMLIT_SERVER_PORT}")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        FRONTEND_SCRIPT,
        "--server.address", STREAMLIT_SERVER_ADDRESS,
        "--server.port", str(STREAMLIT_SERVER_PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    # Run streamlit in the main process so logs appear
    subprocess.run(cmd)

def signal_handler(sig, frame):
    print("\n🛑 Shutting down services...")
    sys.exit(0)

def main():
    print("☁️  Unified Deployment Launcher Starting...")
    print(f"   Backend Port: {API_PORT}")
    print(f"   Frontend Port: {STREAMLIT_SERVER_PORT}")
    print("-" * 40)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start API in a separate thread/process
    # Using a thread with subprocess allows us to capture output if needed
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Wait a moment for API to initialize
    print("⏳ Waiting for API to initialize...")
    time.sleep(3)

    # Start Frontend (this blocks until stopped)
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n👋 Received interrupt, shutting down.")
    except Exception as e:
        print(f"❌ Error running frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
