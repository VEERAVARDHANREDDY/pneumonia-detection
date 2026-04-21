import os
import time
import copy
import random
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
    DATA_DIR = './chest_xray'  # User will place dataset here
    TRAIN_DIR = 'train'
    VAL_DIR = 'val'
    TEST_DIR = 'test'
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 1  # Binary: Normal (0) vs Pneumonia (1)
    
    # Grid Graph Params
    FEAT_MAP_SIZE = 7  # ResNet18 output is 7x7 for 224x224 input
    INPUT_DIM = 512    # ResNet18 feature depth

config = Config()

print(f"Using device: {config.DEVICE}")

# ==========================================
# 1. DATASET PROCESSING
# ==========================================
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, mock_data=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mock_data (bool): If True, generates random noise for testing pipeline logic.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.mock_data = mock_data
        
        if self.mock_data:
            print(f"[{split.upper()}] Generatng MOCK data for testing pipeline...")
            for _ in range(64): # Small number for mock
                self.images.append("mock_path.jpg")
                self.labels.append(random.randint(0, 1))
        else:
            self._load_data()

    def _load_data(self):
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            # Fallback for flat structure or different naming, standard is chest_xray/train
            # Try checking if root_dir itself is the split or contains the classes directly
            print(f"Warning: Directory {split_dir} not found. Checking absolute paths or skipping.")
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
        if self.mock_data:
            # Generate random tensors: 3, 224, 224
            image = torch.rand(3, config.IMG_SIZE, config.IMG_SIZE)
        else:
            img_path = self.images[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a zero tensor if fail
                image = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

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
# 2. FEATURE EXTRACTOR (CNN)
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use ResNet18 pretrained with new weights argument
        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the fully connected layer (fc) and pooling layer (avgpool)
        # We want the spatial features (B, 512, 7, 7)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze early layers if needed, here we simplify and fine-tune all or freeze
        # For this research, let's allow fine-tuning
        
    def forward(self, x):
        # Output: (Batch, 512, 7, 7)
        x = self.features(x)
        return x

# ==========================================
# 3. GRAPH CONSTRUCTION
# ==========================================
def build_adjacency_matrix_grid(size):
    """
    Builds an adjacency matrix for a 2D grid of size x size.
    Interactions: 8-neighborhood (King's move) or 4-neighborhood.
    We will use 8-neighborhood for richer spatial context.
    
    Returns:
        adj: (size*size, size*size) tensor
    """
    num_nodes = size * size
    adj = torch.zeros((num_nodes, num_nodes))
    
    for r in range(size):
        for c in range(size):
            node_idx = r * size + c
            
            # Check all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue # Self
                    
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbor_idx = nr * size + nc
                        adj[node_idx, neighbor_idx] = 1.0
                        
    # Add self-loops (identity matrix) commonly used in GCN
    adj = adj + torch.eye(num_nodes)
    
    # Row-normalize: D^-1 * A
    # Calculate degree
    degree = adj.sum(dim=1)
    d_inv = torch.pow(degree, -1)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat_inv = torch.diag(d_inv)
    
    norm_adj = torch.mm(d_mat_inv, adj)
    return norm_adj

# ==========================================
# 4. GRAPH NEURAL NETWORK (GNN)
# ==========================================
class GCNLayer(nn.Module):
    """
    Simple Graph Convolutional Layer: H_new = Activation( A * H * W )
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj):
        """
        x: Node features (Batch, Num_Nodes, In_Features)
        adj: Adjacency matrix (Num_Nodes, Num_Nodes)
        """
        # 1. Message Passing: Multiply Adjacency with Features
        # x_agg = A * X
        # Since x is batched, we broadcast A
        x_agg = torch.matmul(adj, x) # (Batch, Num_Nodes, In_Features)
        
        # 2. Linear Transformation
        out = self.linear(x_agg)
        
        # 3. Activation
        return self.activation(out)

class GNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super(GNNModule, self).__init__()
        # Precompute adjacency matrix for the fixed grid
        self.num_nodes = num_nodes
        adj = build_adjacency_matrix_grid(int(np.sqrt(num_nodes)))
        # Register as buffer to save state but not update gradients
        self.register_buffer('adj', adj)
        
        # GCN Layers
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Attention Pooling (Global Context) - instead of just mean
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (Batch, Num_Nodes, Input_Dim)
        """
        batch_size = x.size(0)
        adj = self.adj.unsqueeze(0).repeat(batch_size, 1, 1) # Expand for batch
        
        # GCN Pass
        x = self.layer1(x, adj)
        x = self.layer2(x, adj) # (Batch, Num_Nodes, Hidden_Dim)
        
        # Graph-Level Pooling (Readout)
        # Weighted mean based on attention scores
        # attn_weights: (Batch, Num_Nodes, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1) 
        
        # Global representation: Sum(Attn * Features)
        graph_repr = torch.sum(attn_weights * x, dim=1) # (Batch, Hidden_Dim)
        
        return graph_repr, attn_weights

# ==========================================
# 5. HYBRID CNN-GNN MODEL
# ==========================================
class HybridCNNGNN(nn.Module):
    def __init__(self):
        super(HybridCNNGNN, self).__init__()
        self.cnn = FeatureExtractor()
        
        # Feature map 7x7 -> 49 nodes
        self.num_nodes = config.FEAT_MAP_SIZE * config.FEAT_MAP_SIZE
        self.input_dim = config.INPUT_DIM
        
        self.gnn = GNNModule(input_dim=self.input_dim, 
                             hidden_dim=128, 
                             output_dim=1, # Not used since we classify after pooling
                             num_nodes=self.num_nodes)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logits
        )

    def forward(self, x):
        # 1. Extract features: (B, 512, 7, 7)
        features = self.cnn(x)
        
        # 2. Reshape for Graph: (B, 512, 49) -> (B, 49, 512)
        # Treat each pixel as a node
        b, c, h, w = features.size()
        features = features.view(b, c, -1).permute(0, 2, 1)
        
        # 3. Apply GNN
        graph_embedding, attn_weights = self.gnn(features)
        
        # 4. Classification
        logits = self.classifier(graph_embedding)
        
        return logits, attn_weights

# ==========================================
# 6. TRAINING & EVALUATION UTILS
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        # Add tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for inputs, labels in pbar:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.sigmoid(logits) > 0.5
            running_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels).sum().item()
            total += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / total, 'acc': corrects / total})
            
        epoch_loss = running_loss / total
        epoch_acc = corrects / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        if val_loader:
            val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print("Model saved!")
        else:
            print("No validation set provided, skipping validation.")

    return history

def evaluate_model(model, loader, criterion=None):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Use tqdm also for evaluation
    desc = "Validating" if criterion else "Testing"
    pbar = tqdm(loader, desc=desc)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).unsqueeze(1)
            
            logits, _ = model(inputs)
            if criterion:
                loss = criterion(logits, labels)
                running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(logits) > 0.5
            corrects += (preds == labels).sum().item()
            total += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    loss = running_loss / total if criterion else 0.0
    acc = corrects / total
    print(f"Val/Test Loss: {loss:.4f} Acc: {acc:.4f}")
    
    return loss, acc, (all_labels, all_preds)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    # Detect if data exists, otherwise use mock
    use_mock = False
    if not os.path.exists(os.path.join(config.DATA_DIR, config.TRAIN_DIR)):
        print("Dataset not found in default path. Using MOCK data for demonstration.")
        print("Please configure Config.DATA_DIR in the script to point to your 'chest_xray' folder.")
        use_mock = True
    else:
        print("Dataset found! Loading real data...")

    # Load Data
    train_dataset = ChestXRayDataset(config.DATA_DIR, config.TRAIN_DIR, train_transforms, mock_data=use_mock)
    val_dataset = ChestXRayDataset(config.DATA_DIR, config.VAL_DIR, test_transforms, mock_data=use_mock)
    test_dataset = ChestXRayDataset(config.DATA_DIR, config.TEST_DIR, test_transforms, mock_data=use_mock)
    
    # If using mock, datasets might be empty if we rely on file scan, so we force length in mock mode inside class
    # Check if loaded
    if len(train_dataset) == 0 and not use_mock:
        print("Error: No images found. Check your dataset structure.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = HybridCNNGNN().to(config.DEVICE)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train
    print("\nStarting Training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.EPOCHS)
    
    # Plot Training
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']: plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if history['val_acc']: plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('training_history.png')
    print("Training plot saved to training_history.png")
    
    # Final Evaluation
    print("\nLoading Best Model for Testing...")
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    
    _, test_acc, (y_true, y_pred) = evaluate_model(model, test_loader)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    main()
