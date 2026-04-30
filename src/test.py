import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


# =========================
# 1. CONFIGURATION
# =========================
DATA_DIR = Path("data")
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. MULTI-TASK DATASET WRAPPER
# =========================
class MultiTaskImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.base_dataset = datasets.ImageFolder(root=root, transform=transform)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, scene_label = self.base_dataset[idx]

        class_name = self.classes[scene_label]

        # Scene label from folder structure
        scene = scene_label

        # Indoor/Outdoor label
        # Current Intel dataset is mostly outdoor scenes.
        # 0 = indoor, 1 = outdoor
        indoor = 1

        # Day/Night label
        # Current dataset appears mostly daytime.
        # 0 = day, 1 = night
        day = 0

        # Weather label
        # 0 = clear/normal, 1 = cold/snow-like, 2 = water/coastal
        if class_name in ["glacier", "mountain"]:
            weather = 1
        elif class_name in ["sea"]:
            weather = 2
        else:
            weather = 0

        labels = {
            "scene": torch.tensor(scene, dtype=torch.long),
            "indoor": torch.tensor(indoor, dtype=torch.long),
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
        num_scene_classes: int,
        num_indoor_classes: int = 2,
        num_day_classes: int = 2,
        num_weather_classes: int = 3,
        freeze_backbone: bool = False,
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
        self.indoor_head = nn.Linear(in_features, num_indoor_classes)
        self.day_head = nn.Linear(in_features, num_day_classes)
        self.weather_head = nn.Linear(in_features, num_weather_classes)

    def forward(self, x):
        features = self.backbone(x)

        return {
            "scene": self.scene_head(features),
            "indoor": self.indoor_head(features),
            "day": self.day_head(features),
            "weather": self.weather_head(features),
            "features": features,
        }


# =========================
# 5. MAIN EXECUTION
# =========================
def main():
    train_dataset = MultiTaskImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset = MultiTaskImageFolder(DATA_DIR / "val", transform=eval_transforms)
    test_dataset = MultiTaskImageFolder(DATA_DIR / "test", transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    class_names = train_dataset.classes
    num_scene_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of scene classes:", num_scene_classes)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # Check one batch
    images, labels = next(iter(train_loader))
    images = images.to(DEVICE)

    print("\nLabel batch shapes:")
    print("Scene labels:", labels["scene"].shape)
    print("Indoor labels:", labels["indoor"].shape)
    print("Day labels:", labels["day"].shape)
    print("Weather labels:", labels["weather"].shape)

    print("\nExample labels from first batch:")
    print("Scene:", labels["scene"][:10])
    print("Indoor:", labels["indoor"][:10])
    print("Day:", labels["day"][:10])
    print("Weather:", labels["weather"][:10])

    # Test model outputs
    multitask_model = MultiTaskResNet18(
        num_scene_classes=num_scene_classes,
        num_indoor_classes=2,
        num_day_classes=2,
        num_weather_classes=3,
        freeze_backbone=False
    ).to(DEVICE)

    outputs = multitask_model(images)

    print("\nMulti-task model output shapes:")
    print("Scene output:", outputs["scene"].shape)
    print("Indoor/Outdoor output:", outputs["indoor"].shape)
    print("Day/Night output:", outputs["day"].shape)
    print("Weather output:", outputs["weather"].shape)
    print("Feature output:", outputs["features"].shape)

    print("\nMulti-task dataset and model test completed successfully.")


if __name__ == "__main__":
    main()