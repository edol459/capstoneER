# 2_explore_data.py - Data Exploration and Visualization


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import random

print("=" * 70)
print("EMOTION DETECTION - DATA EXPLORATION")
print("=" * 70)

# Configuration
DATA_DIR = "prepared_data"

# Verify data exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"\nError: '{DATA_DIR}' not found!\n"
        f"Please run 1_prepare_data.py first."
    )

# count images
def count_images_per_class(split_dir):
    class_counts = {}
    for emotion in os.listdir(split_dir):
        emotion_dir = os.path.join(split_dir, emotion)
        if os.path.isdir(emotion_dir):
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[emotion] = len(image_files)
    return class_counts

# Get class distributions
train_counts = count_images_per_class(os.path.join(DATA_DIR, "train"))
val_counts = count_images_per_class(os.path.join(DATA_DIR, "val"))
test_counts = count_images_per_class(os.path.join(DATA_DIR, "test"))

emotions = sorted(train_counts.keys())

print("\nClass Distribution:")
print(f"{'Emotion':<15} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-" * 50)
for emotion in emotions:
    print(f"{emotion:<15} {train_counts[emotion]:<10} {val_counts[emotion]:<10} {test_counts[emotion]:<10}")

# ============================================================================
# VISUALIZATION 1: Class Distribution Bar Chart
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (split, counts) in enumerate([("Train", train_counts), 
                                        ("Val", val_counts), 
                                        ("Test", test_counts)]):
    emotions_list = sorted(counts.keys())
    values = [counts[e] for e in emotions_list]
    
    axes[idx].bar(emotions_list, values, color='steelblue', alpha=0.8)
    axes[idx].set_title(f'{split} Set Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Emotion')
    axes[idx].set_ylabel('Number of Images')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VISUALIZATION 2: Sample Images from Each Class
# ============================================================================
print("\nLoading sample images...")
samples_per_class = 3
fig, axes = plt.subplots(len(emotions), samples_per_class, 
                         figsize=(12, 2.5 * len(emotions)))

for row, emotion in enumerate(emotions):
    emotion_dir = os.path.join(DATA_DIR, "train", emotion)
    image_files = [f for f in os.listdir(emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select sample images
    sample_images = random.sample(image_files, min(samples_per_class, len(image_files)))
    
    for col, img_file in enumerate(sample_images):
        img_path = os.path.join(emotion_dir, img_file)
        img = Image.open(img_path)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        
        if col == 0:
            axes[row, col].set_title(f'{emotion}', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VISUALIZATION 3: Image Size Analysis
# ============================================================================
print("\nAnalyzing image dimensions...")
widths = []
heights = []
sample_size = 100  # Sample 100 images for speed

for emotion in emotions:
    emotion_dir = os.path.join(DATA_DIR, "train", emotion)
    image_files = [f for f in os.listdir(emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    samples = random.sample(image_files, min(sample_size // len(emotions), len(image_files)))
    
    for img_file in samples:
        img_path = os.path.join(emotion_dir, img_file)
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)

print(f"\nImage Size Statistics (from {len(widths)} samples):")
print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
print(f"  Most common size: {Counter(zip(widths, heights)).most_common(1)[0][0]}")

# ============================================================================
# VISUALIZATION 4: Pixel Intensity Distribution
# ============================================================================
print("\nAnalyzing pixel intensity distributions...")
pixel_values = []
sample_size = 50

for emotion in emotions[:3]:  # Sample from first 3 emotions
    emotion_dir = os.path.join(DATA_DIR, "train", emotion)
    image_files = [f for f in os.listdir(emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    samples = random.sample(image_files, min(sample_size // 3, len(image_files)))
    
    for img_file in samples:
        img_path = os.path.join(emotion_dir, img_file)
        img = np.array(Image.open(img_path))
        pixel_values.extend(img.flatten())

plt.figure(figsize=(10, 5))
plt.hist(pixel_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.title('Pixel Intensity Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('pixel_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"Total emotion classes: {len(emotions)}")
print(f"Emotion labels: {', '.join(emotions)}")
print(f"Total training images: {sum(train_counts.values())}")
print(f"Total validation images: {sum(val_counts.values())}")
print(f"Total test images: {sum(test_counts.values())}")
print(f"Total images: {sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())}")

# Check class balance
train_values = list(train_counts.values())
balance_ratio = max(train_values) / min(train_values)
print(f"\nClass balance ratio (max/min): {balance_ratio:.2f}")
if balance_ratio > 2:
    print(" Warning: Dataset is imbalanced. Consider using weighted loss or oversampling.")
else:
    print("  Dataset is reasonably balanced.")
