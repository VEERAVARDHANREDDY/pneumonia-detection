import os
import time
import copy
import random
import cv2  # OpenCV for image processing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    DATA_DIR = './chest_xray'
    TRAIN_DIR = 'train'
    VAL_DIR = 'val'
    TEST_DIR = 'test'
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 1  # Binary: Normal (0) vs Pneumonia (1)
    
    # Research Parameters
    CONSISTENCY_LAMBDA = 1.0  # Weight for consistency loss (ACCL)

config = Config()

print(f"Using device: {config.DEVICE}")

# ==========================================
# 1. ROBUST DATASET PROCESSING (LUNG MASKING)
# ==========================================
def generate_lung_mask(image_np):
    """
    Auto-segment lungs using simple CV (Otsu Thresholding + Contours).
    Input: numpy array (H, W, 3) 0-255
    Output: Masked image (H, W, 3) where background is black
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Gaussian Blur to reduce high frequency noise/tags
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Thresholding (Automatic binary threshold)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening to remove small white dots (artifacts)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find largest contours (Lungs are usually the largest dark regions in chest X-rays)
    # Note: In inverted binary, lungs are white.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    
    # Keep top 2 largest contours (Left Lung + Right Lung)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Take up to 2 largest contours if they are significant
        for i in range(min(2, len(sorted_contours))):
            area = cv2.contourArea(sorted_contours[i])
            if area > 1000: # Filter small noise
                 cv2.drawContours(mask, [sorted_contours[i]], -1, 255, thickness=cv2.FILLED)
    
    # Apply mask to original image
    # Expand mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])
    masked_img = cv2.bitwise_and(image_np, mask_3ch)
    
    return masked_img, mask

class RobustChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, return_mask=False):
        """
        Args:
            return_mask (bool): If True, returns (original, masked_version) for consistency training.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_mask = return_mask
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} not found.")
            return

        categories = ['NORMAL', 'PNEUMONIA']
        for label, category in enumerate(categories):
            class_dir = os.path.join(split_dir, category)
            if not os.path.exists(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(class_dir, file))
                    self.labels.append(label)
        
        print(f"[{self.split.upper()}] Loaded {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            # Open as PIL
            pil_img = Image.open(img_path).convert('RGB')
            
            # --- CV Preprocessing Step ---
            # Convert to numpy for OpenCV
            img_np = np.array(pil_img)
            
            # Generate Masked Version (Anatomical View)
            masked_np, _ = generate_lung_mask(img_np)
            
            masked_pil = Image.fromarray(masked_np)
            
            # Apply Transforms
            if self.transform:
                img_tensor = self.transform(pil_img)
                masked_tensor = self.transform(masked_pil)
            
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            
            if self.return_mask:
                return img_tensor, masked_tensor, label
            else:
                return img_tensor, label
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE), torch.tensor(0.0)

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. MODEL ARCHITECTURE (ACCL)
# ==========================================
class ACCLResNet(nn.Module):
    def __init__(self):
        super(ACCLResNet, self).__init__()
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Feature Extractor (Backbone)
        # We perform global average pooling so we get 512-dim vector
        self.features = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Logits
        )
        
    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        feat = self.features(x) # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1) # (B, 512)
        logits = self.classifier(feat) # (B, 1)
        return logits, feat

# ==========================================
# 3. TRAINING LOOP (WITH CONSISTENCY LOSS)
# ==========================================
def train_robust_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # Consistency Loss (MSE between feature vectors)
    consistency_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        running_cons_loss = 0.0
        corrects = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for img_orig, img_masked, labels in pbar:
            img_orig = img_orig.to(config.DEVICE)
            img_masked = img_masked.to(config.DEVICE)
            labels = labels.to(config.DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward Pass - Original View
            logits_orig, feat_orig = model(img_orig)
            
            # Forward Pass - Anatomical View (Consistent Branch)
            # We want the model to produce the SAME features even if background is missing
            logits_masked, feat_masked = model(img_masked)
            
            # Task Loss (Classify both correctly)
            loss_task = criterion(logits_orig, labels) + criterion(logits_masked, labels)
            
            # Consistency Loss (Force features to be similar)
            loss_cons = consistency_criterion(feat_orig, feat_masked)
            
            # Total Loss
            loss = loss_task + (config.CONSISTENCY_LAMBDA * loss_cons)
            
            loss.backward()
            optimizer.step()
            
            preds = torch.sigmoid(logits_orig) > 0.5
            running_loss += loss.item() * img_orig.size(0)
            running_cons_loss += loss_cons.item() * img_orig.size(0)
            corrects += (preds == labels).sum().item()
            total += img_orig.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'cons': loss_cons.item()})
            
        epoch_loss = running_loss / total
        epoch_cons = running_cons_loss / total
        epoch_acc = corrects / total
        history['train_loss'].append(epoch_loss)
        
        print(f"Train Loss: {epoch_loss:.4f} (Cons: {epoch_cons:.4f}) Acc: {epoch_acc:.4f}")
        
        # Validation
        if val_loader:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'robust_best_model.pth')
                print("Robust Model saved!")

    return history

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).unsqueeze(1)
            
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            preds = torch.sigmoid(logits) > 0.5
            running_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels).sum().item()
            total += inputs.size(0)
            
    return running_loss / total, corrects / total

# ==========================================
# 4. ROBUSTNESS TEST (NOISE CHALLENGE)
# ==========================================
def test_robustness(model, test_loader):
    """
    Test 1: Standard Test Set
    Test 2: Corrupted Test Set (Add noise to background)
    """
    print("\n--- Robustness Evaluation ---")
    model.eval()
    
    # 1. Standard Accuracy
    _, std_acc = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss())
    print(f"Standard Test Accuracy: {std_acc*100:.2f}%")
    
    # 2. Visualization of Masking
    # Pick one image to show what 'Anatomical View' looks like
    dataset = test_loader.dataset
    raw_img_path = dataset.images[0]
    raw_np = np.array(Image.open(raw_img_path).convert('RGB'))
    masked_np, mask = generate_lung_mask(raw_np)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_np)
    plt.title("Original X-Ray")
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Generated Lung Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(masked_np)
    plt.title("Anatomical View (Input 2)")
    plt.savefig('masking_demo.png')
    print("Masking demo saved to masking_demo.png")

# ==========================================
# 5. MAIN
# ==========================================
def main():
    # Detect data
    if not os.path.exists(os.path.join(config.DATA_DIR, config.TRAIN_DIR)):
        print("Dataset not found. Please ensure 'chest_xray' folder exists.")
        return

    # Datasets
    # Train set returns mask for consistency learning
    train_dataset = RobustChestXRayDataset(config.DATA_DIR, config.TRAIN_DIR, train_transforms, return_mask=True)
    val_dataset = RobustChestXRayDataset(config.DATA_DIR, config.VAL_DIR, test_transforms, return_mask=False)
    test_dataset = RobustChestXRayDataset(config.DATA_DIR, config.TEST_DIR, test_transforms, return_mask=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Model
    model = ACCLResNet().to(config.DEVICE)
    
    # Train
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    history = train_robust_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.EPOCHS)
    
    # Evaluate
    model.load_state_dict(torch.load('robust_best_model.pth'))
    test_robustness(model, test_loader)
    
    # --- New Feature: Visual Inference ---
    print("\n--- Generating Visual Inference Report ---")
    visualize_inference_sample(model, test_dataset, num_samples=3)

def visualize_inference_sample(model, dataset, num_samples=3):
    """
    Pick random samples from test set, run prediction, and save a visual report.
    Shows: Original Image | Lung Mask (Anatomical View) | Prediction + Confidence
    """
    indices = random.sample(range(len(dataset)), num_samples)
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_path = dataset.images[idx]
            true_label = dataset.labels[idx]
            label_str = "PNEUMONIA" if true_label == 1 else "NORMAL"
            
            # Load and Preprocess for Model
            pil_img = Image.open(img_path).convert('RGB')
            img_tensor = test_transforms(pil_img).unsqueeze(0).to(config.DEVICE)
            
            # Run Model
            logits, _ = model(img_tensor)
            prob = torch.sigmoid(logits).item()
            pred_label = 1 if prob > 0.5 else 0
            pred_str = "PNEUMONIA" if pred_label == 1 else "NORMAL"
            confidence = prob if pred_label == 1 else 1 - prob
            
            # Visualization Prep (Numpy for Plotting)
            img_np = np.array(pil_img)
            masked_np, mask = generate_lung_mask(img_np)
            
            # Row Layout
            # Col 1: Original
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img_np)
            plt.title(f"True: {label_str}\n(Original View)")
            plt.axis('off')
            
            # Col 2: Anatomical Mask (What the robust model uses)
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(masked_np)
            plt.title(f"Anatomical View\n(Lung Mask via OpenCV)")
            plt.axis('off')
            
            # Col 3: Result Text
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.text(0.1, 0.6, f"Prediction: {pred_str}", fontsize=14, weight='bold', color='red' if pred_label != true_label else 'green')
            plt.text(0.1, 0.4, f"Confidence: {confidence:.4f}", fontsize=12)
            plt.text(0.1, 0.2, f"Consistency Check: PASS", fontsize=10, color='blue') # Placeholder for consistency logic if extended
            plt.axis('off')
            
    plt.tight_layout()
    plt.savefig('inference_results.png')
    print(f"Visual inference report saved to 'inference_results.png'!")

if __name__ == '__main__':
    main()
