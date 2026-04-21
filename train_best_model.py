import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from main_pipeline import Config, ChestXRayDataset, test_transforms

# Ensure output directory exists
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_densenet169():
    print("Loading DenseNet169...")
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
            
    return DenseNetFeatureExtractor(model).to(Config.DEVICE)

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            feats = model(inputs)
            feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

def main():
    # 1. Load Data
    print("Loading datasets...")
    train_dataset = ChestXRayDataset(Config.DATA_DIR, Config.TRAIN_DIR, test_transforms)
    test_dataset = ChestXRayDataset(Config.DATA_DIR, Config.TEST_DIR, test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 2. Model
    model = get_densenet169()
    
    # 3. Extract
    print("Extracting Training Features...")
    X_train, y_train = extract_features(model, train_loader, Config.DEVICE)
    
    print("Extracting Test Features...")
    X_test, y_test = extract_features(model, test_loader, Config.DEVICE)
    
    # 4. Train SVM
    print("Training SVM (RBF)...")
    base_clf = SVC(kernel='rbf', probability=True, C=1.0)
    # CalibratedClassifierCV isn't strictly needed if probability=True in SVC, but sometimes better.
    # We'll stick to SVC(probability=True) as per comparison script.
    clf = base_clf
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))
    
    # 6. Save
    save_path = os.path.join(OUTPUT_DIR, "DenseNet169_SVM_RBF.pkl")
    joblib.dump(clf, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
