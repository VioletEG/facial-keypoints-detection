from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from facial_keypoints.data import (
    IMAGE_SIZE,
    FacialKeypointsDataset,
    build_horizontal_flip_mappings,
    load_train_dataframe,
    prepare_train_data,
)
from facial_keypoints.models import MaskedSmoothL1Loss, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse_with_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    se = (((pred - target) * IMAGE_SIZE) ** 2) * mask
    mse = se.sum() / mask.sum().clamp_min(1.0)
    return float(torch.sqrt(mse).item())


def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    images: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    flip_indices: np.ndarray,
    x_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    train_ds = FacialKeypointsDataset(
        images[train_idx],
        targets[train_idx],
        masks[train_idx],
        augment=not args.disable_augment,
        flip_indices=flip_indices,
        x_mask=x_mask,
    )
    val_ds = FacialKeypointsDataset(images[val_idx], targets[val_idx], masks[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, num_outputs=targets.shape[1]).to(device)
    criterion = MaskedSmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )

    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, mb in tqdm(train_loader, desc=f"fold={fold} epoch={epoch} train", leave=False):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_targets, val_masks = [], [], []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu()
                val_preds.append(pred)
                val_targets.append(yb)
                val_masks.append(mb)

        val_pred = torch.cat(val_preds, dim=0)
        val_target = torch.cat(val_targets, dim=0)
        val_mask = torch.cat(val_masks, dim=0)
        val_rmse = rmse_with_mask(val_pred, val_target, val_mask)

        print(
            f"fold={fold} epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6e} "
            f"train_loss={np.mean(train_losses):.6f} val_rmse={val_rmse:.6f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    if best_state is None:
        raise RuntimeError("No best state found during training")
    return best_state, best_rmse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train facial keypoints model with K-Fold CV")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing Kaggle CSV files")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--model", type=str, default="improved", choices=["baseline", "improved"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--disable-augment", action="store_true", help="Disable random horizontal-flip augmentation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = load_train_dataframe(args.data_dir)
    train_data = prepare_train_data(train_df)

    images, targets, masks = train_data.images, train_data.targets, train_data.masks
    flip_indices, x_mask = build_horizontal_flip_mappings(train_data.target_columns)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images), start=1):
        best_state, best_rmse = train_one_fold(
            fold, train_idx, val_idx, images, targets, masks, flip_indices, x_mask, args, device
        )
        fold_scores.append(best_rmse)

        torch.save(
            {
                "model_state": best_state,
                "model_name": args.model,
                "target_columns": train_data.target_columns,
            },
            args.output_dir / f"fold_{fold}.pt",
        )
        print(f"Saved fold_{fold}.pt | best_val_rmse={best_rmse:.6f}")

    summary = {
        "model": args.model,
        "folds": args.folds,
        "fold_scores": fold_scores,
        "cv_mean_rmse": float(np.mean(fold_scores)),
        "cv_std_rmse": float(np.std(fold_scores)),
    }
    with open(args.output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("CV summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
