import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import joblib
import argparse
from main_pipeline import Config

# Re-use the ModelFactory logic or import it. 
# Better to copy-paste the relevant parts for a standalone inference script 
# to avoid dependency hell if model_comparison changes.
# But for now, let's keep it simple and consistent.

class InferenceModel:
    def __init__(self, model_name="DenseNet169", classifier_path="comparison_results/DenseNet169_SVM_RBF.pkl"):
        self.device = Config.DEVICE
        self.model_name = model_name
        
        # Load Backbone
        print(f"Loading backbone: {model_name}...")
        self.backbone = self._load_backbone(model_name)
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Load Classifier
        print(f"Loading classifier from {classifier_path}...")
        if not os.path.exists(classifier_path):
             raise FileNotFoundError(f"Classifier not found at {classifier_path}. Please run model_comparison.py or train_best_model.py first.")
        self.classifier = joblib.load(classifier_path)
        
        # Transforms (Standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_backbone(self, model_name):
        # Simplified loader from model_comparison logic
        if model_name == "DenseNet169":
            model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
            class DenseNetFeatureExtractor(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.features = original_model.features
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                def forward(self, x):
                    features = self.features(x)
                    out = nn.functional.relu(features, inplace=True)
                    out = self.avgpool(out)
                    return out.flatten(1)
            return DenseNetFeatureExtractor(model)
        
        elif model_name == "ResNet50":
             model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
             modules = list(model.children())[:-1]
             return nn.Sequential(*modules)
             
        # Add others if needed
        else:
            raise ValueError("Only DenseNet169 and ResNet50 supported in this demo script currently.")

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.backbone(input_tensor)
                features = features.view(features.size(0), -1).cpu().numpy()
            
            # Classify
            prediction = self.classifier.predict(features)[0]
            probability = self.classifier.predict_proba(features)[0][1] # Probability of class 1 (Pneumonia)
            
            label = "PNEUMONIA" if prediction == 1 else "NORMAL"
            confidence = probability if prediction == 1 else 1 - probability
            
            return label, confidence
            
        except Exception as e:
            return f"Error: {e}", 0.0

def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", type=str, default="DenseNet169", help="Backbone model name")
    parser.add_argument("--clf", type=str, default="comparison_results/DenseNet169_SVM_RBF.pkl", help="Path to classifier model")
    
    args = parser.parse_args()
    
    try:
        inferencer = InferenceModel(args.model, args.clf)
        label, conf = inferencer.predict(args.image)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {conf:.2%}")
    except Exception as e:
        print(f"Failed to initialize: {e}")

if __name__ == "__main__":
    main()
