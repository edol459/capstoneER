# prepare_data.py - Data Preparation for FER Dataset

import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

print("=" * 70)
print("EMOTION DETECTION - DATA PREPARATION")
print("=" * 70)

# Configuration
RAW_TRAIN_DIR = "emotion-detection-fer/train"  # Original train folder
RAW_TEST_DIR = "emotion-detection-fer/test"    # Original test folder
OUTPUT_DIR = "prepared_data"
VAL_SPLIT = 0.15  # Take 15% of train data for validation
RANDOM_STATE = 42   # for reproducibility

# Verify raw data exists
if not os.path.exists(RAW_TRAIN_DIR):
    raise FileNotFoundError(
        f"\nError: Dataset not found at '{RAW_TRAIN_DIR}'\n"
    )

if not os.path.exists(RAW_TEST_DIR):
    raise FileNotFoundError(
        f"\nError: Test set not found at '{RAW_TEST_DIR}'\n"
    )

# Clean up old prepared data
if os.path.exists(OUTPUT_DIR):
    print(f"\nRemoving existing '{OUTPUT_DIR}' directory...")
    shutil.rmtree(OUTPUT_DIR)

# Get emotion classes from train directory
emotion_classes = [d for d in os.listdir(RAW_TRAIN_DIR) 
                   if os.path.isdir(os.path.join(RAW_TRAIN_DIR, d))]
emotion_classes.sort()

print(f"\nFound {len(emotion_classes)} emotion classes: {emotion_classes}")

# Create directory structure
print("\nCreating directory structure...")
for split in ["train", "val", "test"]:
    for emotion in emotion_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

# Split train data into train + validation
print("\nSplitting training data into train/val...")
split_stats = {emotion: {"train": 0, "val": 0, "test": 0} for emotion in emotion_classes}

for emotion in emotion_classes:
    train_emotion_dir = os.path.join(RAW_TRAIN_DIR, emotion)
    
    # Get all image files from original train set
    image_files = [f for f in os.listdir(train_emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"  Warning: No images found for {emotion}")
        continue
    
    # Create full paths
    image_paths = [os.path.join(train_emotion_dir, f) for f in image_files]
    
    # Split into train and validation
    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE
    )
    
    # Copy files to train directory
    for img_path in train_paths:
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, "train", emotion))
    
    # Copy files to validation directory
    for img_path in val_paths:
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, "val", emotion))
    
    split_stats[emotion]["train"] = len(train_paths)
    split_stats[emotion]["val"] = len(val_paths)

# Copy test data
print("\nCopying test data...")
for emotion in emotion_classes:
    test_emotion_dir = os.path.join(RAW_TEST_DIR, emotion)
    
    if not os.path.exists(test_emotion_dir):
        print(f"  Warning: No test directory found for {emotion}")
        continue
    
    # Get all image files from original test set
    image_files = [f for f in os.listdir(test_emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Copy files to test directory
    for img_file in image_files:
        src = os.path.join(test_emotion_dir, img_file)
        dst = os.path.join(OUTPUT_DIR, "test", emotion)
        shutil.copy2(src, dst)
    
    split_stats[emotion]["test"] = len(image_files)

# Print summary statistics
print("\n" + "=" * 70)
print("DATA SPLIT SUMMARY")
print("=" * 70)
print(f"{'Emotion':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
print("-" * 70)

total_train = total_val = total_test = 0
for emotion in emotion_classes:
    train_count = split_stats[emotion]["train"]
    val_count = split_stats[emotion]["val"]
    test_count = split_stats[emotion]["test"]
    total_count = train_count + val_count + test_count
    
    print(f"{emotion:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")
    
    total_train += train_count
    total_val += val_count
    total_test += test_count

print("-" * 70)
print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_train+total_val+total_test:<10}")
print("=" * 70)

# Calculate split percentages
total_images = total_train + total_val + total_test
train_pct = (total_train / total_images) * 100
val_pct = (total_val / total_images) * 100
test_pct = (total_test / total_images) * 100

print(f"\nSplit Percentages:")
print(f"  Train: {train_pct:.1f}%")
print(f"  Val:   {val_pct:.1f}%")
print(f"  Test:  {test_pct:.1f}%")

print(f"\nData preparation complete")
print(f"   Prepared data saved to '{OUTPUT_DIR}/' directory")
