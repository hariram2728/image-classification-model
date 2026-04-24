import torch
import torchvision.models as models
import json
import os
import hashlib
from pathlib import Path

def setup_demo():
    print("🚀 Setting up Secure Demo Environment...")
    
    # 1. Create models directory
    os.makedirs("models", exist_ok=True)
    
    # 2. Download a pre-trained ResNet18 (small & fast)
    print("⬇️  Downloading pre-trained ResNet18 model...")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Save the model
        model_path = "models/demo_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Calculate checksum for integrity verification
        with open(model_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        with open("models/model.sha256", "w") as f:
            f.write(checksum)
        print(f"🔒 Checksum saved: {checksum[:12]}...")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return
    
    # 3. Get ImageNet Class Labels (1000 classes)
    print("🏷️  Generating class labels...")
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        labels = weights.meta["categories"]
        
        # Save labels to JSON
        with open("models/class_labels.json", "w") as f:
            json.dump(labels, f)
        print(f"✅ Labels saved to models/class_labels.json ({len(labels)} classes)")
    except Exception as e:
        print(f"❌ Error saving labels: {e}")

    print("\n✨ Secure demo setup complete! Run 'python api/main.py' to start.")

if __name__ == "__main__":
    setup_demo()
