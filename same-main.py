"""
Emotion Detection
EfficientNet-B0 (Improved Version)
"""

import os
import copy
import random
import time
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm   # IMPROVEMENT: progress bars

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                     ("mps" if torch.backends.mps.is_available() else "cpu"))

IMG_SIZE = 120
EXCLUDE_EMOTIONS = ['disgust']
MAX_SAMPLES_PER_CLASS = None



# DATASET + DATALOADERS


class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, exclude_classes=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        exclude_classes = exclude_classes or []

        # find classes
        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d not in exclude_classes
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # load samples
        self.samples = []
        for emotion in self.classes:
            label = self.class_to_idx[emotion]
            emotion_dir = os.path.join(root_dir, emotion)
            images = [
                (os.path.join(emotion_dir, f), label)
                for f in os.listdir(emotion_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if max_samples_per_class and len(images) > max_samples_per_class:
                random.shuffle(images)
                images = images[:max_samples_per_class]

            # IMPROVEMENT: remove corrupted files BEFORE training
            valid_images = []
            for img_path, lbl in images:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    valid_images.append((img_path, lbl))
                except:
                    print(f"[Warning] Corrupted image removed: {img_path}")

            self.samples.extend(valid_images)

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load safely (should already be valid)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_dir="prepared_data", batch_size=64,
                    exclude_classes=None, max_samples_per_class=None):

    exclude_classes = exclude_classes or []

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = EmotionDataset(os.path.join(data_dir, "train"), train_transform, exclude_classes, max_samples_per_class)
    val_dataset = EmotionDataset(os.path.join(data_dir, "val"), val_test_transform, exclude_classes)
    test_dataset = EmotionDataset(os.path.join(data_dir, "test"), val_test_transform, exclude_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n=====================================")
    print("DATA LOADED")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes


# MODEL


def make_efficientnet_b0(num_classes=6, pretrained=True):
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # patch for grayscale
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # replace classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def create_model(config, num_classes=6):
    return make_efficientnet_b0(
        num_classes=num_classes,
        pretrained=config.get("pretrained", True)
    )


# TRAINING + EARLY STOPPING


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_correct, total_loss, total = 0, 0.0, 0

    # IMPROVEMENT: add tqdm progress bar
    for xb, yb in tqdm(loader, desc="Training", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()

        with autocast("cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_correct, total_loss, total = 0, 0.0, 0
    all_preds, all_targets = [], []

    # IMPROVEMENT: tqdm for validation
    for xb, yb in tqdm(loader, desc="Validating", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        with autocast("cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

        total_correct += (preds == yb).sum().item()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    return total_loss / total, total_correct / total, np.concatenate(all_targets), np.concatenate(all_preds)


def fit(model, train_loader, val_loader, optimizer, epochs, criterion, scheduler=None, scaler=None):

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    patience = 5                      # IMPROVEMENT: Early stopping
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):

        print(f"\nEpoch {ep}/{epochs}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f" train_acc={tr_acc:.4f} | val_acc={val_acc:.4f}")

        # save best model and reset patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  ✓ New best model!")
        else:
            patience_counter += 1

        # IMPROVEMENT: early stopping
        if patience_counter >= patience:
            print("\nEarly stopping activated.")
            break

        if scheduler:
            scheduler.step(val_acc)

    model.load_state_dict(best_model)
    return model, history



# MAIN


def main():

    print("\n" + "="*80)
    print("EfficientNet-B0 Emotion Recognition + AMP")
    print("="*80)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="prepared_data",
        batch_size=64,
        exclude_classes=EXCLUDE_EMOTIONS,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )

    num_classes = len(class_names)

    experiments = [
        {
            "name": "EfficientNetB0-v1",
            "model_type": "efficientnet_b0",
            "lr": 0.0003,
            "optimizer": "Adam",
            "use_scheduler": True,
        }
    ]

    results = [run_experiment(cfg, train_loader, val_loader, num_classes, epochs=25)
               for cfg in experiments]

    best_idx = int(np.argmax([r["val_acc"] for r in results]))
    best = results[best_idx]

    print(f"\nBest Model: {best['config']['name']} with val_acc={best['val_acc']:.4f}")

    # final testing
    y_true, y_pred, y_prob = evaluate_on_test(best["model"], test_loader, num_classes)
    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)

    print("\n===== TEST RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names)

    # save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": best["model"].state_dict(),
        "config": best["config"],
        "class_names": class_names
    }, "models/best_efficientnet_b0_amp.pth")

    print("\nModel saved: models/best_efficientnet_b0_amp.pth")


if __name__ == "__main__":
    main()
