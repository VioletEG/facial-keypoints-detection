from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGE_SIZE = 96


@dataclass(frozen=True)
class TrainData:
    images: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    target_columns: list[str]


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    return (coords / float(IMAGE_SIZE)).astype(np.float32)


def denormalize_coordinates(coords: np.ndarray) -> np.ndarray:
    return (coords * float(IMAGE_SIZE)).astype(np.float32)


def build_horizontal_flip_mappings(target_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build index mapping and x-coordinate mask used for horizontal flips.

    Expected naming follows the Kaggle convention:
    `left_*` and `right_*` columns are treated as mirrored pairs.
    Columns without those prefixes map to themselves.
    """
    col_to_idx = {c: i for i, c in enumerate(target_columns)}
    flip_indices = np.arange(len(target_columns), dtype=np.int64)
    x_mask = np.zeros(len(target_columns), dtype=bool)

    for i, col in enumerate(target_columns):
        if col.startswith("left_"):
            paired = "right_" + col[len("left_") :]
        elif col.startswith("right_"):
            paired = "left_" + col[len("right_") :]
        else:
            paired = col
        flip_indices[i] = col_to_idx.get(paired, i)
        x_mask[i] = col.endswith("_x")

    return flip_indices, x_mask


def apply_horizontal_flip_to_targets(
    targets: np.ndarray,
    flip_indices: np.ndarray,
    x_mask: np.ndarray,
) -> np.ndarray:
    flipped = targets[..., flip_indices].copy()
    flipped[..., x_mask] = 1.0 - flipped[..., x_mask]
    return flipped


def _parse_image(pixel_string: str) -> np.ndarray:
    arr = np.fromiter((float(x) for x in pixel_string.split()), dtype=np.float32)
    if arr.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Unexpected image length {arr.size}, expected {IMAGE_SIZE * IMAGE_SIZE}")
    return (arr.reshape(IMAGE_SIZE, IMAGE_SIZE) / 255.0).astype(np.float32)


def load_train_dataframe(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "training.csv")


def load_test_dataframe(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test.csv")


def load_id_lookup(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "IdLookupTable.csv")


def prepare_train_data(df: pd.DataFrame) -> TrainData:
    target_columns = [c for c in df.columns if c != "Image"]
    images = np.stack(df["Image"].map(_parse_image).values)
    targets = df[target_columns].to_numpy(dtype=np.float32)
    masks = ~np.isnan(targets)
    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)
    targets = normalize_coordinates(targets)
    return TrainData(images=images, targets=targets, masks=masks.astype(np.float32), target_columns=target_columns)


def prepare_test_images(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["Image"].map(_parse_image).values)


class FacialKeypointsDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        augment: bool = False,
        flip_indices: np.ndarray | None = None,
        x_mask: np.ndarray | None = None,
    ) -> None:
        self.images = images
        self.targets = targets
        self.masks = masks
        self.augment = augment
        self.flip_indices = flip_indices
        self.x_mask = x_mask
        if self.targets is not None and self.masks is None:
            raise ValueError("masks must be provided when targets are provided")
        if self.augment and self.targets is None:
            raise ValueError("augment=True requires targets")
        if self.augment and (self.flip_indices is None or self.x_mask is None):
            raise ValueError("augment=True requires flip_indices and x_mask")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.targets is None:
            return torch.from_numpy(image).unsqueeze(0)

        target = self.targets[idx]
        mask = self.masks[idx]

        if self.augment and np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            target = apply_horizontal_flip_to_targets(target, self.flip_indices, self.x_mask)
            mask = mask[self.flip_indices].copy()

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target), torch.from_numpy(mask)
