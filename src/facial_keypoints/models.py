from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, num_outputs: int = 30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ImprovedCNN(nn.Module):
    def __init__(self, num_outputs: int = 30) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.15),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MaskedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff_sq = (pred - target) ** 2
        masked = diff_sq * mask
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_coord = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        masked = per_coord * mask
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom



def build_model(name: str, num_outputs: int) -> nn.Module:
    name = name.lower()
    if name == "baseline":
        return BaselineCNN(num_outputs=num_outputs)
    if name == "improved":
        return ImprovedCNN(num_outputs=num_outputs)
    raise ValueError(f"Unknown model name: {name}")
