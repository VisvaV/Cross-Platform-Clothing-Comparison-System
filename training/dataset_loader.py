import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFashionTripletDataset(Dataset):
    """
    Triplet dataset built from DeepFashion's list_eval_partition.txt.

    For each sample returns (anchor, positive, negative) tensors where:
    - anchor and positive are different images of the same item_id
    - negative is an image from a different item_id
    """

    def __init__(
        self,
        dataset_root: str,
        partition_file: str,
        split: str = "train",
    ):
        """
        Args:
            dataset_root: Root directory prepended to relative image paths
                          (e.g. "DeepFashion" so paths become
                          "DeepFashion/img/WOMEN/...").
            partition_file: Path to list_eval_partition.txt.
            split: One of "train", "val", "test".
        """
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Build item_id -> [image_paths] for the requested split
        item_images: dict[str, list[str]] = defaultdict(list)
        with open(partition_file, "r") as f:
            lines = f.readlines()

        # Line 0: total count, Line 1: header, Line 2+: data
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_path, item_id, status = parts[0], parts[1], parts[2]
            if status == split:
                item_images[item_id].append(img_path)

        # Keep only items with at least 2 images (required for anchor/positive)
        self.item_ids = [
            iid for iid, paths in item_images.items() if len(paths) >= 2
        ]
        self.item_images = {iid: item_images[iid] for iid in self.item_ids}

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int):
        item_id = self.item_ids[idx]
        paths = self.item_images[item_id]

        # Pick two distinct indices for anchor and positive
        anchor_idx, positive_idx = random.sample(range(len(paths)), 2)
        anchor_path = paths[anchor_idx]
        positive_path = paths[positive_idx]

        # Sample a negative from a different item_id
        neg_item_id = item_id
        while neg_item_id == item_id:
            neg_item_id = random.choice(self.item_ids)
        negative_path = random.choice(self.item_images[neg_item_id])

        anchor = self._load(anchor_path)
        positive = self._load(positive_path)
        negative = self._load(negative_path)

        return anchor, positive, negative

    def _load(self, relative_path: str):
        full_path = os.path.join(self.dataset_root, relative_path)
        image = Image.open(full_path).convert("RGB")
        return self.transform(image)
