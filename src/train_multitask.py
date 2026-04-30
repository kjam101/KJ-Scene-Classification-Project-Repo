import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import ImageStat
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# BUILDS UPON BASELINE MODEL

# =========================
# 1. CONFIGURATION
# =========================
DATA_DIR = Path("data")
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FREEZE_BACKBONE = False
ENABLE_PLOTS = True

MODEL_SAVE_PATH = Path("models") / "best_multitask_uncertainty_resnet18.pth"

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

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
# 3. MULTI-TASK DATASET
# =========================
class MultiTaskImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.base_dataset = datasets.ImageFolder(root=root, transform=transform)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_path, scene_label = self.base_dataset.samples[idx]

        original_image = self.base_dataset.loader(image_path).convert("RGB")
        image = self.base_dataset.transform(original_image) if self.base_dataset.transform else original_image

        class_name = self.classes[scene_label]

        scene = scene_label

        if class_name in ["buildings", "street"]:
            environment = 0
        else:
            environment = 1

        grayscale = original_image.convert("L")
        stat = ImageStat.Stat(grayscale)

        brightness = stat.mean[0]
        contrast = stat.stddev[0]

        brightness_threshold = 100
        contrast_threshold = 40

        if brightness < brightness_threshold or contrast < contrast_threshold:
            day = 1
        else:
            day = 0

        if class_name in ["glacier", "mountain"]:
            weather = 1
        elif class_name in ["sea"]:
            weather = 2
        else:
            weather = 0

        labels = {
            "scene": torch.tensor(scene, dtype=torch.long),
            "environment": torch.tensor(environment, dtype=torch.long),
            "day": torch.tensor(day, dtype=torch.long),
            "weather": torch.tensor(weather, dtype=torch.long),
        }

        return image, labels


# =========================
# 4. MULTI-TASK MODEL
# =========================
class MultiTaskResNet18(nn.Module):
    def __init__(
        self,
        num_scene_classes,
        num_environment_classes=2,
        num_day_classes=2,
        num_weather_classes=3,
        freeze_backbone=False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.scene_head = nn.Linear(in_features, num_scene_classes)
        self.environment_head = nn.Linear(in_features, num_environment_classes)
        self.day_head = nn.Linear(in_features, num_day_classes)
        self.weather_head = nn.Linear(in_features, num_weather_classes)

    def forward(self, x):
        features = self.backbone(x)

        return {
            "scene": self.scene_head(features),
            "environment": self.environment_head(features),
            "day": self.day_head(features),
            "weather": self.weather_head(features),
            "features": features,
        }


# =========================
# 5. UNCERTAINTY-WEIGHTED LOSS
# =========================
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses):
        total_loss = 0.0

        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]

        return total_loss


# =========================
# 6. HELPER FUNCTIONS
# =========================
def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == labels).item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, dataloader, criteria, uncertainty_loss_fn, optimizer, device):
    model.train()
    uncertainty_loss_fn.train()

    running_loss = 0.0
    scene_correct = 0
    environment_correct = 0
    day_correct = 0
    weather_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)

        scene_labels = labels["scene"].to(device)
        environment_labels = labels["environment"].to(device)
        day_labels = labels["day"].to(device)
        weather_labels = labels["weather"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        scene_loss = criteria["scene"](outputs["scene"], scene_labels)
        environment_loss = criteria["environment"](outputs["environment"], environment_labels)
        day_loss = criteria["day"](outputs["day"], day_labels)
        weather_loss = criteria["weather"](outputs["weather"], weather_labels)

        loss = uncertainty_loss_fn([
            scene_loss,
            environment_loss,
            day_loss,
            weather_loss
        ])

        loss.backward()
        optimizer.step()

        batch_size = scene_labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        c, _ = compute_accuracy(outputs["scene"], scene_labels)
        scene_correct += c

        c, _ = compute_accuracy(outputs["environment"], environment_labels)
        environment_correct += c

        c, _ = compute_accuracy(outputs["day"], day_labels)
        day_correct += c

        c, _ = compute_accuracy(outputs["weather"], weather_labels)
        weather_correct += c

    return {
        "loss": running_loss / total_samples,
        "scene_acc": scene_correct / total_samples,
        "environment_acc": environment_correct / total_samples,
        "day_acc": day_correct / total_samples,
        "weather_acc": weather_correct / total_samples,
    }


def evaluate(model, dataloader, criteria, uncertainty_loss_fn, device):
    model.eval()
    uncertainty_loss_fn.eval()

    running_loss = 0.0
    scene_correct = 0
    environment_correct = 0
    day_correct = 0
    weather_correct = 0
    total_samples = 0

    scene_preds = []
    scene_true = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            scene_labels = labels["scene"].to(device)
            environment_labels = labels["environment"].to(device)
            day_labels = labels["day"].to(device)
            weather_labels = labels["weather"].to(device)

            outputs = model(images)

            scene_loss = criteria["scene"](outputs["scene"], scene_labels)
            environment_loss = criteria["environment"](outputs["environment"], environment_labels)
            day_loss = criteria["day"](outputs["day"], day_labels)
            weather_loss = criteria["weather"](outputs["weather"], weather_labels)

            loss = uncertainty_loss_fn([
                scene_loss,
                environment_loss,
                day_loss,
                weather_loss
            ])

            batch_size = scene_labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            c, _ = compute_accuracy(outputs["scene"], scene_labels)
            scene_correct += c

            c, _ = compute_accuracy(outputs["environment"], environment_labels)
            environment_correct += c

            c, _ = compute_accuracy(outputs["day"], day_labels)
            day_correct += c

            c, _ = compute_accuracy(outputs["weather"], weather_labels)
            weather_correct += c

            _, preds = torch.max(outputs["scene"], 1)
            scene_preds.extend(preds.cpu().tolist())
            scene_true.extend(scene_labels.cpu().tolist())

    return {
        "loss": running_loss / total_samples,
        "scene_acc": scene_correct / total_samples,
        "environment_acc": environment_correct / total_samples,
        "day_acc": day_correct / total_samples,
        "weather_acc": weather_correct / total_samples,
        "scene_preds": scene_preds,
        "scene_true": scene_true,
    }


# =========================
# 7. VISUALIZATION FUNCTIONS
# =========================
def plot_training_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_scene_acc"], label="Train Scene Accuracy")
    plt.plot(epochs, history["val_scene_acc"], label="Validation Scene Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Multi-Task Scene Classification Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "multitask_accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multi-Task Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "multitask_loss_curve.png")
    plt.close()


def plot_confusion_matrix_from_preds(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Multi-Task Scene Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================
# 8. MAIN
# =========================
def main():
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = MultiTaskImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset = MultiTaskImageFolder(DATA_DIR / "val", transform=eval_transforms)
    test_dataset = MultiTaskImageFolder(DATA_DIR / "test", transform=eval_transforms)

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
    num_scene_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of scene classes:", num_scene_classes)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Device:", DEVICE)
    print("Freeze backbone:", FREEZE_BACKBONE)
    print("Auxiliary tasks: environment type, day/night, weather")

    model = MultiTaskResNet18(
        num_scene_classes=num_scene_classes,
        num_environment_classes=2,
        num_day_classes=2,
        num_weather_classes=3,
        freeze_backbone=FREEZE_BACKBONE
    ).to(DEVICE)

    criteria = {
        "scene": nn.CrossEntropyLoss(),
        "environment": nn.CrossEntropyLoss(),
        "day": nn.CrossEntropyLoss(),
        "weather": nn.CrossEntropyLoss(),
    }

    uncertainty_loss_fn = MultiTaskUncertaintyLoss(num_tasks=4).to(DEVICE)

    optimizer = optim.Adam(
        list(model.parameters()) + list(uncertainty_loss_fn.parameters()),
        lr=LEARNING_RATE
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_wts = copy.deepcopy(uncertainty_loss_fn.state_dict())
    best_val_scene_acc = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_scene_acc": [],
        "val_scene_acc": [],
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 45)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criteria,
            uncertainty_loss_fn,
            optimizer,
            DEVICE
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criteria,
            uncertainty_loss_fn,
            DEVICE
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_scene_acc"].append(train_metrics["scene_acc"])
        history["val_scene_acc"].append(val_metrics["scene_acc"])

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Scene Acc: {train_metrics['scene_acc']:.4f} | "
            f"Env Acc: {train_metrics['environment_acc']:.4f} | "
            f"Day Acc: {train_metrics['day_acc']:.4f} | "
            f"Weather Acc: {train_metrics['weather_acc']:.4f}"
        )

        print(
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Scene Acc: {val_metrics['scene_acc']:.4f} | "
            f"Env Acc: {val_metrics['environment_acc']:.4f} | "
            f"Day Acc: {val_metrics['day_acc']:.4f} | "
            f"Weather Acc: {val_metrics['weather_acc']:.4f}"
        )

        print(
            f"Auxiliary Val Accuracies - "
            f"Environment Type: {val_metrics['environment_acc']:.4f}, "
            f"Day/Night: {val_metrics['day_acc']:.4f}, "
            f"Weather: {val_metrics['weather_acc']:.4f}"
        )

        print("Learned log variances:", uncertainty_loss_fn.log_vars.detach().cpu().numpy())

        if val_metrics["scene_acc"] > best_val_scene_acc:
            best_val_scene_acc = val_metrics["scene_acc"]
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss_wts = copy.deepcopy(uncertainty_loss_fn.state_dict())

            torch.save({
                "model_state_dict": model.state_dict(),
                "loss_state_dict": uncertainty_loss_fn.state_dict(),
                "class_names": class_names,
                "freeze_backbone": FREEZE_BACKBONE,
                "brightness_threshold": 100,
                "contrast_threshold": 40,
                "environment_mapping": {
                    "built_urban": ["buildings", "street"],
                    "natural_outdoor": ["forest", "glacier", "mountain", "sea"],
                }
            }, MODEL_SAVE_PATH)

            print(f"Best uncertainty-weighted multi-task model saved to {MODEL_SAVE_PATH}")

    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed / 60:.2f} minutes")
    print(f"Best validation scene accuracy: {best_val_scene_acc:.4f}")

    model.load_state_dict(best_model_wts)
    uncertainty_loss_fn.load_state_dict(best_loss_wts)

    test_metrics = evaluate(
        model,
        test_loader,
        criteria,
        uncertainty_loss_fn,
        DEVICE
    )

    print("\nTest Results")
    print("-" * 45)
    print(f"Test Loss:             {test_metrics['loss']:.4f}")
    print(f"Scene Test Acc:        {test_metrics['scene_acc']:.4f}")
    print(f"Environment Test Acc:  {test_metrics['environment_acc']:.4f}")
    print(f"Day/Night Test Acc:    {test_metrics['day_acc']:.4f}")
    print(f"Weather Test Acc:      {test_metrics['weather_acc']:.4f}")

    if ENABLE_PLOTS:
        plot_training_history(history, PLOTS_DIR)
        print("Multi-task accuracy and loss curves saved.")

        plot_confusion_matrix_from_preds(
            test_metrics["scene_true"],
            test_metrics["scene_preds"],
            class_names,
            PLOTS_DIR / "multitask_confusion_matrix.png"
        )
        print("Multi-task confusion matrix saved.")

    print("\nFinal learned log variances:")
    print(uncertainty_loss_fn.log_vars.detach().cpu().numpy())

    print("\nAblation comparison targets:")
    print("Baseline Scene Test Acc:                  ~0.9333")
    print("Manual-weight Multi-task Scene Test Acc:   0.9350")
    print(f"Uncertainty Multi-task Scene Test Acc:     {test_metrics['scene_acc']:.4f}")


if __name__ == "__main__":
    main()