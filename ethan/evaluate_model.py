# ============================================================================
# 5_evaluate_model.py - Model Evaluation (48x48 grayscale)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

print("=" * 70)
print("EMOTION DETECTION - MODEL EVALUATION")
print("=" * 70)

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
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# MODEL ARCHITECTURE 
# ============================================================================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
print("\nLoading model and data...")

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Load test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset = EmotionDataset('prepared_data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")

# Load model
checkpoint = torch.load('models/best_model.pth', map_location=device)
num_classes = len(test_dataset.classes)

model = EmotionCNN(num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(" Model loaded successfully")
print(f"Best validation accuracy from training: {checkpoint['val_acc']:.2f}%")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\nGenerating predictions on test set...")

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
print("\nGenerating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_dataset.classes, 
            yticklabels=test_dataset.classes,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes,
            ax=axes[1], cbar_kws={'label': 'Proportion'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(" Saved: logs/confusion_matrix.png")
plt.show()

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(all_labels, all_preds, 
                               target_names=test_dataset.classes,
                               digits=4)
print(report)

# Save to file
with open('logs/classification_report.txt', 'w') as f:
    f.write(report)
print("Saved: logs/classification_report.txt")

# ============================================================================
# PER-CLASS ACCURACY
# ============================================================================
print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

class_accuracies = []
for i, emotion in enumerate(test_dataset.classes):
    class_mask = all_labels == i
    if class_mask.sum() > 0:
        class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() * 100
        class_accuracies.append(class_acc)
        print(f"{emotion:<15} {class_acc:>6.2f}%")

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(test_dataset.classes, class_accuracies, color='steelblue', alpha=0.8)
plt.axhline(y=np.mean(class_accuracies), color='red', linestyle='--', 
            label=f'Average: {np.mean(class_accuracies):.2f}%', linewidth=2)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('logs/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("\naved: logs/per_class_accuracy.png")
plt.show()

# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

# Get confidence scores (max probability for each prediction)
confidences = np.max(all_probs, axis=1)
correct_mask = all_preds == all_labels

correct_confidences = confidences[correct_mask]
incorrect_confidences = confidences[~correct_mask]

print(f"Average confidence (correct predictions): {correct_confidences.mean():.4f}")
print(f"Average confidence (incorrect predictions): {incorrect_confidences.mean():.4f}")

# Plot confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(correct_confidences, bins=30, alpha=0.7, color='green', edgecolor='black', label='Correct')
axes[0].hist(incorrect_confidences, bins=30, alpha=0.7, color='red', edgecolor='black', label='Incorrect')
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Box plot
data_to_plot = [correct_confidences, incorrect_confidences]
axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
axes[1].set_ylabel('Confidence Score', fontsize=12)
axes[1].set_title('Confidence Score Comparison', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('logs/confidence_analysis.png', dpi=300, bbox_inches='tight')
print(" Saved: logs/confidence_analysis.png")
plt.show()

# ============================================================================
# SAMPLE PREDICTIONS
# ============================================================================
print("\nGenerating sample predictions visualization...")

# Get some correct and incorrect predictions
correct_indices = np.where(correct_mask)[0][:6]
incorrect_indices = np.where(~correct_mask)[0][:6] if (~correct_mask).sum() >= 6 else np.where(~correct_mask)[0]

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('Sample Predictions (48x48 Grayscale)', fontsize=16, fontweight='bold')

# Denormalize function
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Show correct predictions
for idx, sample_idx in enumerate(correct_indices):
    img_path, true_label = test_dataset.samples[sample_idx]
    img = Image.open(img_path).convert('L')
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(probs).item()
        confidence = probs[0, pred_label].item()
    
    img_display = denormalize(img_tensor.squeeze().cpu()).numpy()
    
    axes[0, idx].imshow(img_display, cmap='gray')
    axes[0, idx].set_title(f'✓ {test_dataset.classes[pred_label]}\n({confidence:.2f})', 
                          fontsize=10, color='green')
    axes[0, idx].axis('off')

# Show incorrect predictions
for idx, sample_idx in enumerate(incorrect_indices):
    img_path, true_label = test_dataset.samples[sample_idx]
    img = Image.open(img_path).convert('L')
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(probs).item()
        confidence = probs[0, pred_label].item()
    
    img_display = denormalize(img_tensor.squeeze().cpu()).numpy()
    
    axes[1, idx].imshow(img_display, cmap='gray')
    axes[1, idx].set_title(f'✗ Pred: {test_dataset.classes[pred_label]}\nTrue: {test_dataset.classes[true_label]}\n({confidence:.2f})', 
                          fontsize=9, color='red')
    axes[1, idx].axis('off')

# Fill empty subplots if not enough incorrect predictions
for idx in range(len(incorrect_indices), 6):
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('logs/sample_predictions.png', dpi=300, bbox_inches='tight')
print(" Saved: logs/sample_predictions.png")
plt.show()

# ============================================================================
# SAVE EVALUATION METRICS
# ============================================================================
overall_accuracy = (all_preds == all_labels).mean() * 100

evaluation_metrics = {
    'overall_accuracy': float(overall_accuracy),
    'per_class_accuracy': {emotion: float(acc) for emotion, acc in zip(test_dataset.classes, class_accuracies)},
    'average_confidence_correct': float(correct_confidences.mean()),
    'average_confidence_incorrect': float(incorrect_confidences.mean()),
    'num_test_samples': len(test_dataset),
    'num_correct': int((all_preds == all_labels).sum()),
    'num_incorrect': int((all_preds != all_labels).sum())
}

with open('logs/evaluation_metrics.json', 'w') as f:
    json.dump(evaluation_metrics, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"Correctly classified: {(all_preds == all_labels).sum()}/{len(all_labels)}")
print(f"\nGenerated files:")
print(f"   logs/confusion_matrix.png")
print(f"   logs/classification_report.txt")
print(f"   logs/per_class_accuracy.png")
print(f"   logs/confidence_analysis.png")
print(f"   logs/sample_predictions.png")
print(f"   logs/evaluation_metrics.json")
print("=" * 70)