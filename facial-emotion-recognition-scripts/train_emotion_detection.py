#!/usr/bin/env python3

"""
Emotion Detection Model Architecture & Training using FER-2013 Dataset

"""

import os
import copy
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if torch.backends.mps.is_available() else "cpu"))

# Global configuration
IMG_SIZE = 48  # FER-2013 uses 48x48 grayscale images
EXCLUDE_EMOTIONS = ['disgust']  # Exclude disgust
MAX_SAMPLES_PER_CLASS = None  # Limit samples per emotion if desired


# PART A: DATA & SETUP

class AddGaussianNoise:
    """Add Gaussian noise to image tensor for data augmentation"""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class EmotionDataset(Dataset):
    """
    Custom Dataset for FER-2013 emotion detection images.
    Loads 48x48 grayscale images from organized directories.
    """
    def __init__(self, root_dir, transform=None, exclude_classes=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.exclude_classes = exclude_classes or []
        self.max_samples_per_class = max_samples_per_class
        
        # Get emotion classes  excluding specified classes
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))
                              and d not in self.exclude_classes])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image file paths and labels
        self.samples = []
        for emotion in self.classes:
            emotion_dir = os.path.join(root_dir, emotion)
            label = self.class_to_idx[emotion]
            
            # Collect all images for this emotion
            emotion_images = []
            for img_file in os.listdir(emotion_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, img_file)
                    emotion_images.append((img_path, label))
            
            # Limit samples if specified
            if max_samples_per_class and len(emotion_images) > max_samples_per_class:
                random.shuffle(emotion_images)
                emotion_images = emotion_images[:max_samples_per_class]
            
            self.samples.extend(emotion_images)
        
        # Shuffle all samples
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(data_dir="prepared_data", batch_size=64, exclude_classes=None, max_samples_per_class=None):
    """
    Creates train/val/test DataLoaders for FER-2013 emotion detection.
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size for DataLoaders
        exclude_classes: List of emotion classes to exclude
        max_samples_per_class: Maximum samples per emotion (for balancing/speed)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    exclude_classes = exclude_classes or []
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = EmotionDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform,
        exclude_classes=exclude_classes,
        max_samples_per_class=max_samples_per_class
    )
    val_dataset = EmotionDataset(
        os.path.join(data_dir, 'val'),
        transform=val_test_transform,
        exclude_classes=exclude_classes,
        max_samples_per_class=max_samples_per_class // 5 if max_samples_per_class else None
    )
    test_dataset = EmotionDataset(
        os.path.join(data_dir, 'test'),
        transform=val_test_transform,
        exclude_classes=exclude_classes,
        max_samples_per_class=max_samples_per_class // 3 if max_samples_per_class else None
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\n{'='*70}")
    print("DATA LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Emotion classes: {train_dataset.classes}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Excluded classes: {exclude_classes}")
    print(f"Max samples per class: {max_samples_per_class if max_samples_per_class else 'No limit'}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} grayscale")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


# MODEL ARCHITECTURES 

# Initial Simple CNN

class SimpleCNN(nn.Module):
    """
    Initial simple CNN for emotion detection, not using currently
    """
    def __init__(self, num_classes=6, depth=2, base_channels=16, dropout=0.25):
        super(SimpleCNN, self).__init__()
        
        self.depth = depth
        self.base_channels = base_channels
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        in_channels = 1
        ch = base_channels
        
        for i in range(depth):
            layers.append(nn.Conv2d(in_channels, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = ch
            ch = min(ch * 2, 128)  # Cap at 128 channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate final spatial dimensions: 48 / (2^depth)
        final_spatial = 48 // (2 ** depth)
        final_channels = in_channels
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * final_spatial * final_spatial, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


#RESNET ARCHITECTURE- currently using
def make_resnet(num_classes=6, pretrained=True, freeze_backbone=True):
    """
    Create a ResNet18 model adapted for grayscale FER images.
    Args:
        num_classes: number of emotion classes
        pretrained: use pretrained ImageNet weights
        freeze_backbone: whether to freeze earlier layers for feature extraction
    """
    model = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Modify first conv layer to accept 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Replace final  layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # freeze all backbone layers except the classifier
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
                
    return model

# PART C: TRAINING & EVALUATION FUNCTIONS

def train_one_epoch(model, loader, criterion, optimizer):
    """Train model for one epoch"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * yb.size(0)
    
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        
        total += yb.size(0)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * yb.size(0)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    return avg_loss, avg_acc, y_true, y_pred


def fit(model, train_loader, val_loader, optimizer, epochs=15, criterion=None):
    """Train model with early stopping based on validation accuracy"""
    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model)
    return model, history



# EXPERIMENT RUNNER

def run_experiment(config, train_loader, val_loader, num_classes, epochs=20):
    """Run a single experiment with given configuration"""
    model_type = config.get('model_type', 'simplecnn')

    # 1. Model setup
    if model_type == 'resnet18':
        model = make_resnet(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=config.get('freeze_backbone', True)
        ).to(DEVICE)
    else:
        model = SimpleCNN(
            num_classes=num_classes,
            depth=config.get('depth', 2),
            base_channels=config.get('base_channels', 16),
            dropout=config.get('dropout', 0.25)
        ).to(DEVICE)

    #  Compute class weights for balanced loss
    from sklearn.utils.class_weight import compute_class_weight

    # Extract labels from training dataset
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]

    # Compute class weights (higher weight for minority classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_labels
    )

    # Convert to tensor and send to GPU/CPU
    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    print(f"Using class-weighted loss: {weights.cpu().numpy().round(2)}")

    # Weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer
    optimizer_name = config.get('optimizer', 'Adam')
    lr = config.get('lr', 0.0001)
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Train model
    model, history = fit(model, train_loader, val_loader, optimizer, epochs, criterion)
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Return results summary
    return {
        'config': config,
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'num_params': num_params,
        'history': history,
        'model': model,
        'y_true': y_true,
        'y_pred': y_pred,
    }

# EVALUATION & VISUALIZATION

def summarize_results(results):
    """Create summary table of all experiments"""
    rows = []
    for r in results:
        cfg = r['config']
        rows.append({
            'Experiment': cfg.get('name', 'Unknown'),
            'Depth': cfg.get('depth', 2),
            'Base Channels': cfg.get('base_channels', 16),
            'Dropout': cfg.get('dropout', 0.25),
            'Optimizer': cfg.get('optimizer', 'Adam'),
            'Learning Rate': cfg.get('lr', 0.001),
            'Val Loss': f"{r['val_loss']:.4f}",
            'Val Accuracy': f"{r['val_acc']:.4f}",
            'Parameters': r['num_params']
        })
    
    df = pd.DataFrame(rows)
    print("\n" + "="*120)
    print("EXPERIMENTS SUMMARY TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")
    
    return df


def evaluate_on_test(model, test_loader):
    """Evaluate best model on test set"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = accuracy_score(y_true, y_pred)
    
    return acc, y_true, y_pred


def plot_confusion(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('emotion_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrix saved as 'emotion_confusion_matrix.png'")


def plot_training_history(history, title="Training History"):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emotion_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training history saved as 'emotion_training_history.png'")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("EMOTION DETECTION - FER-2013 DATASET")
    print(f"{'='*70}")
    print(f"Using device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} grayscale")
    print(f"Excluding emotions: {EXCLUDE_EMOTIONS}")
    
    # Load data with balanced sampling
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="prepared_data",
        batch_size=64,
        exclude_classes=EXCLUDE_EMOTIONS,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    num_classes = len(class_names)
    
    # Define experiments - simpler, faster configurations
    experiments = [
        # Baseline SimpleCNN experiments
        #{'name': 'Baseline-D2-16ch-Adam-0.001', 'model_type': 'simplecnn', 'depth': 2, 'base_channels': 16,
        #'dropout': 0.25, 'optimizer': 'Adam', 'lr': 0.001},
        
        #{'name': 'Deep-D3-16ch-SGD-0.01', 'model_type': 'simplecnn', 'depth': 3, 'base_channels': 16,
        #'dropout': 0.25, 'optimizer': 'SGD', 'lr': 0.01},
        
        #{'name': 'Wide-D2-32ch-Adam-0.005', 'model_type': 'simplecnn', 'depth': 2, 'base_channels': 32,
        #'dropout': 0.3, 'optimizer': 'Adam', 'lr': 0.005},

        # Try ResNet18
        {'name': 'ResNet18-FullFinetune-Adam-0.0001', 
        'model_type': 'resnet18',
        'optimizer': 'Adam',
        'lr': 0.0001,
        'freeze_backbone': False}
    ]

    
    results = []
    
    for cfg in experiments:
        print(f"\n{'='*80}")
        print(f"🔬 Running experiment: {cfg.get('name', 'exp')}")
        print(f"{'='*80}")
        r = run_experiment(cfg, train_loader, val_loader, num_classes, epochs=20)
        results.append(r)
    
    # Summary table
    df = summarize_results(results)
    
    # Choose best model by validation accuracy
    best_idx = int(np.argmax([r['val_acc'] for r in results]))
    best_model = results[best_idx]['model']
    best_config = results[best_idx]['config']
    
    print(f"\n🏆 Best Model: {best_config.get('name', 'Unknown')}")
    print(f"📊 Best Validation Accuracy: {results[best_idx]['val_acc']:.4f}")
    
    # Plot training history for best model
    plot_training_history(
        results[best_idx]['history'],
        title=f"Training History - {best_config.get('name', 'Best Model')}"
    )
    
    # Test evaluation
    print(f"\n{'='*80}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*80}")
    test_acc, y_true, y_pred = evaluate_on_test(best_model, test_loader)
    print(f"\nTest Accuracy (best model): {test_acc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    plot_confusion(cm, class_names, title="FER-2013 Emotion Detection - Confusion Matrix (Test)")
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_emotion_model.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': best_config,
        'class_names': class_names,
        'test_acc': test_acc
    }, model_path)
    print(f"\nBest model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Files saved: emotion_confusion_matrix.png, emotion_training_history.png")
    print("="*80)


if __name__ == "__main__":
    main()
