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
    return TrainData(images=images, targets=targets, masks=masks.astype(np.float32), target_columns=target_columns)



def prepare_test_images(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["Image"].map(_parse_image).values)


class FacialKeypointsDataset(Dataset):
    def __init__(self, images: np.ndarray, targets: np.ndarray | None = None, masks: np.ndarray | None = None) -> None:
        self.images = images
        self.targets = targets
        self.masks = masks
        if self.targets is not None and self.masks is None:
            raise ValueError("masks must be provided when targets are provided")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)
        if self.targets is None:
            return image
        target = torch.from_numpy(self.targets[idx])
        mask = torch.from_numpy(self.masks[idx])
        return image, target, mask
