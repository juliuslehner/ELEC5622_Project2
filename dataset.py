import os
import random
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# ImageNet normalization constants for pretrained CNNs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 6 cell classes mapped to 0-5
CLASS_NAMES = ["Centromere", "Golgi", "Homogeneous", "NuMem", "Nucleolar", "Speckled"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class DiscreteRotation:
    """Rotates an image by random multiples of 18 degrees on the fly (Gao et al.)."""
    def __init__(self, step=18):
        self.angles = list(range(0, 360, step))

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)


def get_transforms(is_train=True):
    """Returns preprocessing transforms. Augmentation (rotation) only runs on training."""
    ops = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: ImageOps.autocontrast(img))  # Normalizes uneven microscope lighting
    ]

    if is_train:
        ops.append(DiscreteRotation(step=18))

    ops.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transforms.Compose(ops)


class HEp2Dataset(Dataset):
    def __init__(self, folder_path, csv_path="gt_training.csv", is_train=False):
        self.folder_path = folder_path
        self.transform = get_transforms(is_train=is_train)

        # 1. Read CSV and map 5-digit filename -> label index
        df = pd.read_csv(csv_path)
        id_col = df.columns[0]
        class_col = df.columns[1]

        label_map = {}
        for _, row in df.iterrows():
            filename = f"{int(row[id_col]):05d}.png"
            label_map[filename] = CLASS_TO_IDX[str(row[class_col]).strip()]

        # 2. Gather image paths physically present in this split folder
        self.image_paths = []
        self.labels = []

        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".png") and fname in label_map:
                self.image_paths.append(os.path.join(folder_path, fname))
                self.labels.append(label_map[fname])

        print(f"Loaded {len(self.image_paths)} images from {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load one image lazily on demand
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_dataloaders(data_dir="data", csv_path="gt_training.csv", batch_size=64, num_workers=2):
    train_set = HEp2Dataset(os.path.join(data_dir, "training"),   csv_path, is_train=True)
    val_set   = HEp2Dataset(os.path.join(data_dir, "validation"), csv_path, is_train=False)
    test_set  = HEp2Dataset(os.path.join(data_dir, "test"),       csv_path, is_train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, test_loader = get_dataloaders(num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")