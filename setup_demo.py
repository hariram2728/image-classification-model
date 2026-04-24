import torch
import torchvision.models as models
import json
import os

def setup_demo():
    print("🚀 Setting up Demo Environment...")
    os.makedirs("models", exist_ok=True)
    
    # Explicitly use ResNet18
    print("⬇️ Downloading ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Save state dict only
    torch.save(model.state_dict(), "models/demo_model.pth")
    print("✅ Model saved.")
    
    # Save labels
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    labels = weights.meta["categories"]
    with open("models/class_labels.json", "w") as f:
        json.dump(labels, f)
    print("✅ Labels saved.")

if __name__ == "__main__":
    setup_demo()
