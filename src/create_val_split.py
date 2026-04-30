from pathlib import Path
import random
import shutil

# =========================
# CONFIG
# =========================
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

SPLIT_RATIO = 0.2  # 20% to validation
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================
# FUNCTION
# =========================
def create_validation_split():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]

    for class_folder in class_folders:
        class_name = class_folder.name
        images = list(class_folder.glob("*"))

        if len(images) == 0:
            print(f"Skipping empty class: {class_name}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Number to move
        num_val = int(len(images) * SPLIT_RATIO)

        # Create val class folder
        val_class_dir = VAL_DIR / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Move files
        val_images = images[:num_val]

        for img_path in val_images:
            dest = val_class_dir / img_path.name
            shutil.move(str(img_path), str(dest))

        print(f"{class_name}: moved {num_val} images to validation")

    print("\nValidation split complete.")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_validation_split()