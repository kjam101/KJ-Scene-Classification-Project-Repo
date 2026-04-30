import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# CONFIG
# =========================
DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "best_multitask_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# MODEL
# =========================
class MultiTaskResNet18(torch.nn.Module):
    def __init__(self, num_scene_classes):
        super().__init__()

        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

        self.backbone = backbone
        self.scene_head = torch.nn.Linear(in_features, num_scene_classes)

    def forward(self, x):
        features = self.backbone(x)
        return features


# =========================
# MAIN
# =========================
def main():
    dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    model = MultiTaskResNet18(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model = model.to(DEVICE)
    model.eval()

    all_features = []
    all_labels = []

    print("\nExtracting features...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            features = model(images)
            features = features.cpu().numpy()

            all_features.append(features)
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Feature shape:", all_features.shape)

    # =========================
    # UMAP REDUCTION
    # =========================
    print("\nRunning UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_features)

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        idxs = all_labels == i
        plt.scatter(
            embedding[idxs, 0],
            embedding[idxs, 1],
            label=class_names[i],
            alpha=0.6
        )

    plt.legend()
    plt.title("UMAP Visualization of Scene Features")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.savefig("results/plots/umap_plot.png")
    plt.show()

    print("\nUMAP plot saved to results/plots/umap_plot.png")


if __name__ == "__main__":
    main()