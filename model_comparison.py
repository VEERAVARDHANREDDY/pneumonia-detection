import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import joblib
from tqdm import tqdm

# Import configuration and dataset from main pipeline
from main_pipeline import Config, ChestXRayDataset, train_transforms, test_transforms

# Ensure output directory exists
OUTPUT_DIR = "comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ModelFactory:
    @staticmethod
    def get_feature_extractor(model_name):
        device = Config.DEVICE
        print(f"Loading {model_name}...")
        
        if model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(model.children())[:-1] # Remove FC
            model = nn.Sequential(*modules)
            feature_dim = 2048
            
        elif model_name == "VGG16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # VGG Classifier block is distinct, we only want the features block + pooling
            # But standard VGG flatten is huge (7*7*512). Let's use adaptive pool to get 1x1
            model.classifier = nn.Identity() # Remove classifier
            # We usually need to handle the output shape manually for VGG if we want a vector
            # Let's wrap it
            class VGGFeatureExtractor(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.features = original_model.features
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    return x.flatten(1)
            model = VGGFeatureExtractor(model)
            feature_dim = 512
            
        elif model_name == "VGG19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            class VGGFeatureExtractor(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    self.features = original_model.features
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    return x.flatten(1)
            model = VGGFeatureExtractor(model)
            feature_dim = 512

        elif model_name == "DenseNet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Identity()
            # DenseNet features are (B, 1024, 7, 7) before classifier's pooling? 
            # Actually torchvision densenet 'features' returns the map. The forward pass does ReLU -> AvgPool -> Flatten -> FC
            # We need to replicate that
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
            model = DenseNetFeatureExtractor(model)
            feature_dim = 1024

        elif model_name == "DenseNet169":
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
            model = DenseNetFeatureExtractor(model)
            feature_dim = 1664 # DenseNet169 output
            
        else:
            raise ValueError(f"Model {model_name} not supported")
            
        return model.to(device), feature_dim

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            feats = model(inputs)
            # Flatten if not already
            feats = feats.view(feats.size(0), -1)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

def plot_roc_curves(results, model_name):
    plt.figure(figsize=(10, 8))
    for clf_name, metrics in results.items():
        if 'fpr' in metrics and 'tpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], label=f"{clf_name} (AUC = {metrics['auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name} Features')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png"))
    plt.close()

def plot_confusion_matrices(results, model_name):
    num_clfs = len(results)
    cols = 2
    rows = (num_clfs + 1) // 2
    
    plt.figure(figsize=(12, 5 * rows))
    for i, (clf_name, metrics) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"{clf_name} (Acc: {metrics['acc']:.4f})")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    plt.suptitle(f"Confusion Matrices - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrices_{model_name}.png"))
    plt.close()

def main():
    # 1. Setup Data
    # Use config from main_pipeline
    # Check if data exists, if not use mock
    use_mock = False
    if not os.path.exists(os.path.join(Config.DATA_DIR, Config.TRAIN_DIR)):
        print("Dataset not found. Using MOCK data.")
        use_mock = True

    train_dataset = ChestXRayDataset(Config.DATA_DIR, Config.TRAIN_DIR, test_transforms, mock_data=use_mock) # Use test_transforms for standard resizing without augmentation for feature extraction? usually yes, or train_transforms. Let's use test_transforms for consistency in features.
    test_dataset = ChestXRayDataset(Config.DATA_DIR, Config.TEST_DIR, test_transforms, mock_data=use_mock)

    if len(train_dataset) == 0 and not use_mock:
         print("No training data found.")
         return

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Define Models and Classifiers
    model_names = ["ResNet50", "VGG16", "DenseNet121", "DenseNet169"]
    
    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', probability=True, C=1.0),
        "Naive Bayes": GaussianNB(),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    all_results = []

    # 3. Loop
    for m_name in model_names:
        print(f"\n{'='*20}\nProcessing {m_name}...\n{'='*20}")
        
        # Load backbone
        try:
            model, feat_dim = ModelFactory.get_feature_extractor(m_name)
        except Exception as e:
            print(f"Skipping {m_name} due to error: {e}")
            continue

        # Extract features
        print(f"Extracting Training Features for {m_name}...")
        X_train, y_train = extract_features(model, train_loader, Config.DEVICE)
        
        print(f"Extracting Test Features for {m_name}...")
        X_test, y_test = extract_features(model, test_loader, Config.DEVICE)

        # Evaluate Classifiers
        model_results = {}
        
        for clf_name, clf in classifiers.items():
            print(f"Training {clf_name}...")
            clf.fit(X_train, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred # basic fallback
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            model_results[clf_name] = {
                'acc': acc, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 
                'cm': cm, 'fpr': fpr, 'tpr': tpr
            }
            
            all_results.append({
                'Model': m_name,
                'Classifier': clf_name,
                'Accuracy': acc,
                'AUC': auc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1
            })
            
            # Save model
            model_filename = f"{m_name}_{clf_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
            joblib.dump(clf, os.path.join(OUTPUT_DIR, model_filename))

        # Generate Plots for this backbone
        plot_roc_curves(model_results, m_name)
        plot_confusion_matrices(model_results, m_name)

    # 4. Save Overall Results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nComparison completed. Results saved to {csv_path}")
    print(df)
    
    # Plot Comparison Bars
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Model', y='AUC', hue='Classifier')
    plt.title('AUC Score Comparison')
    plt.ylim(0.5, 1.05)
    plt.savefig(os.path.join(OUTPUT_DIR, "auc_comparison_bar.png"))
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Model', y='Accuracy', hue='Classifier')
    plt.title('Accuracy Score Comparison')
    plt.ylim(0.5, 1.05)
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison_bar.png"))

if __name__ == "__main__":
    main()
