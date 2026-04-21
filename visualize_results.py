import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
from main_pipeline import HybridCNNGNN, Config  # Import model structure

# Use the same config
config = Config()

def load_trained_model(model_path):
    model = HybridCNNGNN()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    return image, tensor

def visualize_prediction(model, image_path, true_label=None):
    original_img, tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        logits, attn_weights = model(tensor)
        prob = torch.sigmoid(logits).item()
        pred_label = "Pneumonia" if prob > 0.5 else "Normal"
        confidence = prob if prob > 0.5 else 1 - prob
        
        # Process Attention Weights for Visualization
        # attn_weights shape: (1, 49, 1) -> (49,)
        attn = attn_weights.squeeze().cpu().numpy()
        
        # Reshape to 7x7 grid (corresponding to 7x7 feature map)
        attn_grid = attn.reshape(7, 7)
        
        # Upsample attention map to image size for overlay
        # We use interpolation to smooth it out
        import cv2
        # Normalize attention map to 0-1 for heatmap
        attn_norm = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min())
        attn_resized = cv2.resize(attn_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
        
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # 1. Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    title = f"Pred: {pred_label} ({confidence:.2f})"
    if true_label:
        title += f"\nTrue: {true_label}"
    plt.title(title, color='green' if pred_label == true_label or true_label is None else 'red')
    plt.axis('off')
    
    # 2. Attention Heatmap Overlay (Explainability)
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.imshow(attn_resized, cmap='jet', alpha=0.4) # Overlay heatmap
    plt.title("GNN Attention Map\n(Where the model is looking)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists('best_model.pth'):
        print("Error: 'best_model.pth' not found. Please train the model first.")
        return

    print("Loading model...")
    model = load_trained_model('best_model.pth')
    
    # Pick random images from test set
    test_dir = os.path.join(config.DATA_DIR, config.TEST_DIR)
    categories = ['NORMAL', 'PNEUMONIA']
    
    print("Press CLOSE on the image window to see the next prediction. Ctrl+C to exit.")
    
    try:
        while True:
            # Randomly select a category and an image
            cat = random.choice(categories)
            cat_dir = os.path.join(test_dir, cat)
            if not os.path.exists(cat_dir): continue
            
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if not files: continue
            
            img_name = random.choice(files)
            img_path = os.path.join(cat_dir, img_name)
            
            print(f"Visualizing: {cat}/{img_name}...")
            visualize_prediction(model, img_path, true_label=cat.title())
            
    except KeyboardInterrupt:
        print("\nExiting visualization.")

if __name__ == '__main__':
    main()
