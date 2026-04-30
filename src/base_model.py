import os
import numpy as np
from pathlib import Path
import copy
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================
# 1. CONFIGURATION
# =========================
DATA_DIR = Path("data")  # expects data/train, data/val, data/test
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Use 0 on Windows to avoid multiprocessing issues
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = Path("models") / "best_baseline_resnet18.pth"
# Config for Results
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# ImageNet normalization for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# 2. TRANSFORMS
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =========================
# 3. TRAIN / VALIDATE FUNCTIONS
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += torch.sum(preds == labels).item()
            total_samples += batch_size

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc, all_preds, all_labels


# =========================
# 4. VISUALIZATION FUNCTIONS
# =========================
def plot_training_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


def plot_confusion_matrix_from_preds(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def unnormalize_image(img_tensor, mean, std):
    img = img_tensor.clone().cpu().numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def save_sample_predictions(model, dataloader, class_names, device, output_path, mean, std, num_images=6):
    model.eval()
    images_shown = 0

    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                ax = plt.subplot(2, 3, images_shown + 1)
                img = unnormalize_image(images[i], mean, std)
                ax.imshow(img)

                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                ax.set_title(f"True: {true_label}\nPred: {pred_label}")
                ax.axis("off")

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =========================
# 5. MULTI-TASK MODEL
# =========================
class MultiTaskResNet18(nn.Module):
    def __init__(
        self,
        num_scene_classes: int,
        num_indoor_classes: int = 2,
        num_day_classes: int = 2,
        num_weather_classes: int = 3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load pretrained ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # Get number of features before final FC layer
        in_features = backbone.fc.in_features

        # Remove original classifier so backbone outputs features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Optional: freeze backbone for faster experimentation
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Task-specific heads
        self.scene_head = nn.Linear(in_features, num_scene_classes)
        self.indoor_head = nn.Linear(in_features, num_indoor_classes)
        self.day_head = nn.Linear(in_features, num_day_classes)
        self.weather_head = nn.Linear(in_features, num_weather_classes)

    def forward(self, x):
        features = self.backbone(x)

        outputs = {
            "scene": self.scene_head(features),
            "indoor": self.indoor_head(features),
            "day": self.day_head(features),
            "weather": self.weather_head(features),
            "features": features,  # useful later for UMAP
        }

        return outputs


# =========================
# 6. MAIN EXECUTION
# =========================
def main():
    # Ensure output directories exist
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 7. DATASETS AND DATALOADERS
    # =========================
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transforms)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # =========================
    # 7. MODEL SETUP
    # =========================
    # Use ResNet-18 first for speed. You can switch to ResNet-50 later.
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================
    # 8. TRAINING LOOP
    # =========================
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved to {MODEL_SAVE_PATH}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best weights before test evaluation
    model.load_state_dict(best_model_wts)

    # =========================
    # 9. TEST EVALUATION
    # =========================
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)

    print("\nTest Results")
    print("-" * 30)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    # Save training history plots
    plot_training_history(history, PLOTS_DIR)
    print("Training curves saved.")

    # Save confusion matrix
    plot_confusion_matrix_from_preds(
        test_labels,
        test_preds,
        class_names,
        PLOTS_DIR / "confusion_matrix.png"
    )
    print("Confusion matrix saved.")

    # Save sample predictions
    save_sample_predictions(
        model,
        test_loader,
        class_names,
        DEVICE,
        PREDICTIONS_DIR / "sample_predictions.png",
        IMAGENET_MEAN,
        IMAGENET_STD,
        num_images=6
    )
    print("Sample predictions saved.")

    # =========================
    # 10. OPTIONAL: SAVE CLASS NAMES
    # =========================
    with open("class_names.txt", "w") as f:
        for idx, class_name in enumerate(class_names):
            f.write(f"{idx}: {class_name}\n")

    print("Class names saved to class_names.txt")


if __name__ == "__main__":
    main()