"""
ImageNet-1K data loading for BARRIER/InTAct training on Diversi50 and Confuse5 concepts.

Uses local ImageNet directory (standard ILSVRC2012 format):
    {imagenet_root}/train/n01440764/*.JPEG
    {imagenet_root}/train/n02123045/*.JPEG

Provides forget/remain dataloaders compatible with intact_unlearn training loop.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def get_transform(image_size: int = 512):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


def build_imagenet_class_index(imagenet_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map concept names (lowercase human-readable) to ImageNet class indices."""
    train_dir = os.path.join(imagenet_root, "train")
    ds = ImageFolder(train_dir)
    idx_to_classname: Dict[int, str] = {}
    for class_dir, class_idx in sorted(ds.class_to_idx.items()):
        idx_to_classname[class_idx] = class_dir

    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    categories = {i: name.lower() for i, name in enumerate(weights.meta["categories"])}

    name_to_idx = {}
    for idx in range(1000):
        name = categories.get(idx, "")
        if name:
            name_to_idx[name] = idx

    return name_to_idx, categories


DIVERSI50_CONCEPTS = [
    "tabby", "Labrador retriever", "tiger", "lion", "African elephant",
    "sports car", "convertible", "school bus", "airliner", "mountain bike",
    "minivan", "pickup", "motor scooter",
    "folding chair", "rocking chair", "desk", "dining table", "table lamp",
    "acoustic guitar", "grand piano", "violin", "cornet", "sax",
    "cellular telephone", "reflex camera", "laptop", "television", "computer keyboard",
    "Granny Smith", "orange", "banana", "strawberry", "broccoli", "cauliflower",
    "cowboy hat", "running shoe", "sweatshirt", "jean", "trench coat",
    "pizza", "hotdog", "cheeseburger", "ice cream", "burrito", "mashed potato",
    "traffic light", "backpack", "umbrella", "bookcase", "water bottle",
]

CONFUSE5_PAIRS = [
    ("golden retriever", "labrador retriever"),
    ("tabby", "tiger cat"),
    ("orange", "lemon"),
    ("speedboat", "lifeboat"),
    ("soccer ball", "volleyball"),
]

CONFUSE5_CONCEPTS = [c for pair in CONFUSE5_PAIRS for c in pair]

CONCEPT_OVERRIDES = {
    "jean": "jean, blue jeans, denim",
    "Granny Smith": "granny smith",
    "sweatshirt": "sweatshirt",
    "motor scooter": "motor scooter, scooter",
    "Labrador retriever": "labrador retriever",
    "African elephant": "african elephant, loxodonta africana",
    "reflex camera": "reflex camera",
    "cellular telephone": "cellular telephone, cellular phone, cellphone, cell, mobile phone",
    "trench coat": "trench coat",
    "mashed potato": "mashed potato",
    "acoustic guitar": "acoustic guitar",
    "grand piano": "grand piano, grand",
    "mountain bike": "mountain bike, all-terrain bike, off-roader",
    "traffic light": "traffic light, traffic signal, stoplight",
    "running shoe": "running shoe",
    "cowboy hat": "cowboy hat, ten-gallon hat",
    "ice cream": "ice cream, icecream",
    "dining table": "dining table, board",
    "table lamp": "table lamp",
    "computer keyboard": "computer keyboard, keypad",
    "folding chair": "folding chair",
    "rocking chair": "rocking chair, rocker",
    "water bottle": "water bottle",
    "tiger": "tiger",
    "lion": "lion, king of beasts, panthera leo",
    "school bus": "school bus",
    "airliner": "airliner",
    "hotdog": "hotdog, hot dog, red hot",
    "cheeseburger": "cheeseburger",
    "burrito": "burrito",
    "cornet": "cornet, horn, trumpet, trump",
    "sax": "sax, saxophone",
}


class ImageNetClassSubset(Dataset):
    """Returns (image, label) pairs from ImageNet for a specific class index."""

    def __init__(self, imagenet_root: str, class_idx: int, transform=None):
        full_ds = ImageFolder(os.path.join(imagenet_root, "train"), transform=transform)
        self.indices = [i for i, (_, lbl) in enumerate(full_ds.samples) if lbl == class_idx]
        self.full_ds = full_ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_ds[self.indices[idx]]


IMAGENET_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"


def ensure_imagenet(imagenet_root: str):
    """
    Download and extract ImageNet-1K train split into:
        {imagenet_root}/train/n01440764/*.JPEG
        {imagenet_root}/train/n02123045/*.JPEG
    """
    import subprocess
    import tarfile

    train_dir = os.path.join(imagenet_root, "train")
    has_class_dirs = os.path.isdir(train_dir) and any(
        os.path.isdir(os.path.join(train_dir, e.name))
        for e in os.scandir(train_dir)
    )
    if has_class_dirs:
        print(f"ImageNet-1K found at {imagenet_root}")
        return

    tar_path = os.path.join(imagenet_root, "ILSVRC2012_img_train.tar")
    os.makedirs(imagenet_root, exist_ok=True)

    if os.path.exists(tar_path) and os.path.getsize(tar_path) == 0:
        print("Existing tar archive is empty, re-downloading...")
        os.remove(tar_path)

    if not os.path.exists(tar_path):
        print(f"Downloading ImageNet-1K from {IMAGENET_URL} (~138GB)...")
        subprocess.check_call(
            ["wget", "-c", "-O", tar_path, IMAGENET_URL],
        )

    if os.path.getsize(tar_path) == 0:
        raise RuntimeError(
            f"Downloaded tar archive is empty ({tar_path}). "
            f"ImageNet requires manual download from {IMAGENET_URL} with an account."
        )

    # Check if main archive was already extracted (class .tar files exist in train/)
    class_tars = [f for f in os.listdir(train_dir)
                  if f.endswith(".tar") and os.path.isfile(os.path.join(train_dir, f))]
    if len(class_tars) < 50:
        print("Extracting main archive...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=train_dir, filter="data")
    else:
        print(f"Main archive already extracted ({len(class_tars)} class tarballs found).")

    print("Unpacking class tarballs (this may take a while)...")
    for fname in sorted(os.listdir(train_dir)):
        fpath = os.path.join(train_dir, fname)
        if fname.endswith(".tar"):
            class_dir = fname[:-4]
            os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
            with tarfile.open(fpath) as tf:
                tf.extractall(path=os.path.join(train_dir, class_dir), filter="data")
            os.remove(fpath)

    os.remove(tar_path)
    n_classes = sum(1 for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    print(f"ImageNet-1K is ready ({n_classes} classes).")


def make_forget_remain_dataloaders(
    imagenet_root: str,
    forget_concepts: List[str],
    batch_size: int,
    image_size: int = 512,
    bounds_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create forget (target concept) and remain (all other ImageNet classes)
    dataloaders compatible with intact_unlearn's training loop.

    Returns: (forget_dl, remain_dl, descriptions)
      descriptions[i] = prompt string for class index i
    """
    train_dir = os.path.join(imagenet_root, "train")
    train_dir_path = Path(train_dir)

    if not train_dir_path.exists() or not any(train_dir_path.iterdir()):
        ensure_imagenet(imagenet_root)

    transform = get_transform(image_size)
    name_to_idx, categories = build_imagenet_class_index(imagenet_root)

    forget_indices = []
    for concept in forget_concepts:
        key = concept.lower()
        if key in name_to_idx:
            forget_indices.append(name_to_idx[key])
        elif concept in CONCEPT_OVERRIDES:
            for part in CONCEPT_OVERRIDES[concept].split(","):
                k = part.strip().lower()
                if k in name_to_idx:
                    forget_indices.append(name_to_idx[k])
                    break
        else:
            import re
            for cat_name, cat_idx in name_to_idx.items():
                if re.search(r'\b' + re.escape(key) + r'\b', cat_name):
                    forget_indices.append(cat_idx)
                    break

    if not forget_indices:
        raise ValueError(f"No ImageNet class indices found for concepts: {forget_concepts}")

    f_set = set(forget_indices)
    full_ds = ImageFolder(os.path.join(imagenet_root, "train"), transform=transform)

    f_samples = [(path, lbl) for path, lbl in full_ds.samples if lbl in f_set]
    r_samples = [(path, lbl) for path, lbl in full_ds.samples if lbl not in f_set]

    # Map labels to descriptive prompt strings
    descriptions = [""] * 1000
    for idx, name in categories.items():
        descriptions[idx] = f"an image of a {name}"

    class ForgetDS(Dataset):
        def __init__(self, samples, full_dataset):
            self.samples = samples
            self.full_ds = full_dataset

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, lbl = self.samples[idx]
            full_idx = full_ds.samples.index((path, lbl))
            return self.full_ds[full_idx]

    fds = ForgetDS(f_samples, full_ds)
    rds = ForgetDS(r_samples, full_ds)

    if bounds_fraction < 1.0:
        n_f = max(1, int(len(fds) * bounds_fraction))
        n_r = max(1, int(len(rds) * bounds_fraction))
        fds = Subset(fds, list(range(min(n_f, len(fds)))))
        rds = Subset(rds, list(range(min(n_r, len(rds)))))

    f_dl = DataLoader(fds, batch_size=batch_size, shuffle=True)
    r_dl = DataLoader(rds, batch_size=batch_size, shuffle=True)

    print(f"Forget: {len(fds)} images across classes {sorted(f_set)}")
    print(f"Remain: {len(rds)} images across {1000 - len(f_set)} classes")
    return f_dl, r_dl, descriptions


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/datasets/ImageNet"
    name_to_idx, _ = build_imagenet_class_index(root)
    for concept in DIVERSI50_CONCEPTS[:5]:
        idx = name_to_idx.get(concept.lower(), "NOT FOUND")
        print(f"  {concept:30s} -> {idx}")