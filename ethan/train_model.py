# ============================================================================
# 4_train_model.py - Model Training (48x48 grayscale)
# Run this FOURTH to train the emotion detection model
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import json
from PIL import Image
import os

# ============================================================================
# DATASET CLASS
# ============================================================================
class EmotionDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for emotion in self.classes:
            emotion_dir = os.path.join(root_dir, emotion)
            label = self.class_to_idx[emotion]
            for img_file in os.listdir(emotion_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(emotion_dir, img_file), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

# ============================================================================
# CUSTOM CNN MODEL FOR 48x48 GRAYSCALE IMAGES
# ============================================================================
class EmotionCNN(nn.Module):
    """
    Custom CNN architecture designed for 48x48 grayscale emotion detection
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 48x48x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 24x24x64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 12x12x128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 6x6x256
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 24x24
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 12x12
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> 6x6
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> 3x3
        
        # Flatten and fully connected
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

print("=" * 70)
print("EMOTION DETECTION - MODEL TRAINING")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'img_size': 48,
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'model_name': 'EmotionCNN',
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'num_workers': 0,
    'save_dir': 'models',
    'log_dir': 'logs'
}

# Create directories
os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)

print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Validation/test transforms (no augmentation)
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create datasets
train_dataset = EmotionDataset('prepared_data/train', transform=train_transform)
val_dataset = EmotionDataset('prepared_data/val', transform=val_test_transform)
test_dataset = EmotionDataset('prepared_data/test', transform=val_test_transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                       shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=False)

# ============================================================================
# MODEL SETUP
# ============================================================================
print("\n" + "=" * 70)
print("MODEL SETUP")
print("=" * 70)

device = torch.device(CONFIG['device'])
num_classes = len(train_dataset.classes)

# Create custom CNN model
model = EmotionCNN(num_classes=num_classes)
model = model.to(device)

print(f"\nModel: {CONFIG['model_name']}")
print(f"Device: {device}")
print(f"Number of classes: {num_classes}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress update every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
start_time = time.time()

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-" * 70)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_acc)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes
        }, best_model_path)
        print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")

training_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Training completed in {training_time/60:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Save final model
final_model_path = os.path.join(CONFIG['save_dir'], 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to: {final_model_path}")

# ============================================================================
# TEST EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

# Load best model
checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================
history_path = os.path.join(CONFIG['log_dir'], 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"\nTraining history saved to: {history_path}")

# ============================================================================
# PLOT TRAINING CURVES
# ============================================================================
print("\nGenerating training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
axes[1].plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(CONFIG['log_dir'], 'training_curves.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f" Saved: {plot_path}")
plt.show()

# ============================================================================
# SAVE TRAINING SUMMARY
# ============================================================================
summary = {
    'config': CONFIG,
    'best_val_acc': float(best_val_acc),
    'final_test_acc': float(test_acc),
    'training_time_minutes': training_time / 60,
    'num_train_samples': len(train_dataset),
    'num_val_samples': len(val_dataset),
    'num_test_samples': len(test_dataset),
    'classes': train_dataset.classes,
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_path = os.path.join(CONFIG['log_dir'], 'training_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print(f"Training history: {history_path}")
print(f"Training summary: {summary_path}")
print(f"Training curves: {plot_path}")
print("\nNext step: Run 5_evaluate_model.py for detailed evaluation")
print("=" * 70)