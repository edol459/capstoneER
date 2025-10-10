# ============================================================================
# 3_dataloader.py - Data Loader Implementation (48x48 grayscale)
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("EMOTION DETECTION - DATA LOADER IMPLEMENTATION")
print("=" * 70)

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
class EmotionDataset(Dataset):
    """
    Custom Dataset for emotion detection images.
    
    Args:
        root_dir: Path to data directory (e.g., 'prepared_data/train')
        transform: Torchvision transforms to apply to images
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get emotion classes (subdirectories)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build list of (image_path, label) tuples
        self.samples = []
        for emotion in self.classes:
            emotion_dir = os.path.join(root_dir, emotion)
            label = self.class_to_idx[emotion]
            
            for img_file in os.listdir(emotion_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_dir, img_file)
                    self.samples.append((img_path, label))
    
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
    
    def get_class_name(self, label_idx):
        """Convert label index to emotion name"""
        return self.classes[label_idx]


# ============================================================================
# CUSTOM TRANSFORMS
# ============================================================================
class AddGaussianNoise:
    """Add Gaussian noise to image tensor for data augmentation"""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


# ============================================================================
# DATA AUGMENTATION & PREPROCESSING PIPELINE
# ============================================================================
IMG_SIZE = 48  # Original dataset size

# Training transforms with data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
    AddGaussianNoise(mean=0., std=0.05),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Validation/Test transforms 
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================================
# CREATE DATASETS
# ============================================================================
print("\nCreating datasets...")
train_dataset = EmotionDataset('prepared_data/train', transform=train_transform)
val_dataset = EmotionDataset('prepared_data/val', transform=val_test_transform)
test_dataset = EmotionDataset('prepared_data/test', transform=val_test_transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Emotion classes: {train_dataset.classes}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE} grayscale")

# ============================================================================
# CREATE DATALOADERS
# ============================================================================
BATCH_SIZE = 64
NUM_WORKERS = 0  # Set to 0 for macOS/Windows compatibility

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

print(f"\nBatch size: {BATCH_SIZE}")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# TEST DATA LOADING
# ============================================================================
print("\n" + "=" * 70)
print("TESTING DATA LOADER")
print("=" * 70)

# Load one batch
images, labels = next(iter(train_loader))
print(f"\nBatch shape: {images.shape}")
print(f"Labels shape: {labels.shape}") 
print(f"Image dtype: {images.dtype}")
print(f"Labels dtype: {labels.dtype}")
print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")

# ============================================================================
# VISUALIZE AUGMENTED SAMPLES
# ============================================================================
print("\nGenerating visualization of augmented samples...")

def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]"""
    return tensor * 0.5 + 0.5

# Show multiple augmentations of the same image
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
fig.suptitle('Data Augmentation Examples (same image, different augmentations)', 
             fontsize=14, fontweight='bold')

# Get one image from dataset
sample_img_path, sample_label = train_dataset.samples[0]
sample_img_pil = Image.open(sample_img_path).convert('L')
emotion_name = train_dataset.get_class_name(sample_label)

for i in range(12):
    row = i // 6
    col = i % 6
    
    # Apply transform (creates different augmentation each time)
    augmented = train_transform(sample_img_pil)
    augmented = denormalize(augmented)
    
    # Convert to numpy for display
    img_np = augmented.squeeze().numpy()
    
    axes[row, col].imshow(img_np, cmap='gray')
    axes[row, col].axis('off')
    if i == 0:
        axes[row, col].set_title(f'{emotion_name}', fontsize=10)

plt.tight_layout()
plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
print("Saved: augmentation_examples.png")
plt.show()

# ============================================================================
# VISUALIZE BATCH SAMPLES
# ============================================================================
print("\nGenerating batch visualization...")
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample Training Batch (48x48 grayscale)', fontsize=14, fontweight='bold')

images_to_show = images[:32]  # Show up to 32 images
labels_to_show = labels[:32]

for idx in range(min(32, len(images_to_show))):
    row = idx // 8
    col = idx % 8
    
    img = denormalize(images_to_show[idx])
    img_np = img.squeeze().numpy()
    
    emotion_name = train_dataset.get_class_name(labels_to_show[idx].item())
    
    axes[row, col].imshow(img_np, cmap='gray')
    axes[row, col].set_title(emotion_name, fontsize=8)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('batch_samples.png', dpi=300, bbox_inches='tight')
print(" Saved: batch_samples.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DATA LOADER SUMMARY")
print("=" * 70)
print(f"\nKey Features Implemented:")
print(f"  • Original 48x48 grayscale format preserved")
print(f"  • Random horizontal flip")
print(f"  • Random rotation (±10°)")
print(f"  • Random translation (±10%)")
print(f"  • Gaussian noise")
print(f"  • Normalization to [-1, 1]")
print("\n" + "=" * 70)

train_class_counts = train_dataset.get_class_counts()
for idx, class_name in enumerate(train_dataset.classes):
    print(f"{class_name}: {train_class_counts[idx]} images")
