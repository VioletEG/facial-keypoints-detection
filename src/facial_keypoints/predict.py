from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from facial_keypoints.data import (
    FacialKeypointsDataset,
    IMAGE_SIZE,
    apply_horizontal_flip_to_targets,
    build_horizontal_flip_mappings,
    denormalize_coordinates,
    load_id_lookup,
    load_test_dataframe,
    prepare_test_images,
)
from facial_keypoints.models import build_model

MIN_COORDINATE = 0.0
MAX_COORDINATE = float(IMAGE_SIZE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Kaggle submission for facial keypoints detection")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing Kaggle CSV files")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Directory with fold checkpoints")
    p.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def load_checkpoints(artifacts_dir: Path) -> list[Path]:
    ckpts = sorted(artifacts_dir.glob("fold_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No fold checkpoints found in {artifacts_dir}")
    return ckpts


def predict_with_ensemble(ckpt_paths: list[Path], images: np.ndarray, batch_size: int) -> tuple[np.ndarray, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FacialKeypointsDataset(images)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ensemble = None
    target_columns: list[str] | None = None

    for ckpt_path in ckpt_paths:
        payload = torch.load(ckpt_path, map_location=device)
        model_name = payload["model_name"]
        this_target_cols = payload["target_columns"]
        target_columns = this_target_cols if target_columns is None else target_columns

        model = build_model(model_name, num_outputs=len(this_target_cols)).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        flip_indices, x_mask = build_horizontal_flip_mappings(this_target_cols)

        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                pred_flip = model(torch.flip(xb, dims=[3])).cpu().numpy()
                pred_flip = apply_horizontal_flip_to_targets(pred_flip, flip_indices, x_mask)
                preds.append((pred + pred_flip) * 0.5)

        fold_pred = np.concatenate(preds, axis=0)
        ensemble = fold_pred if ensemble is None else ensemble + fold_pred

    if ensemble is None or target_columns is None:
        raise RuntimeError("No valid checkpoints loaded for ensemble prediction")
    ensemble = ensemble / len(ckpt_paths)
    ensemble = denormalize_coordinates(ensemble)
    return np.clip(ensemble, MIN_COORDINATE, MAX_COORDINATE), target_columns


def build_submission(lookup: pd.DataFrame, pred_matrix: np.ndarray, target_columns: list[str]) -> pd.DataFrame:
    col_to_idx = {c: i for i, c in enumerate(target_columns)}
    locations = []
    for _, row in lookup.iterrows():
        image_id = int(row["ImageId"]) - 1
        feature_name = row["FeatureName"]
        if feature_name not in col_to_idx:
            raise KeyError(f"Feature {feature_name} missing in trained targets")
        val = float(pred_matrix[image_id, col_to_idx[feature_name]])
        locations.append(val)

    return pd.DataFrame({"RowId": lookup["RowId"].astype(int), "Location": locations})


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    test_df = load_test_dataframe(args.data_dir)
    lookup = load_id_lookup(args.data_dir)
    test_images = prepare_test_images(test_df)

    ckpt_paths = load_checkpoints(args.artifacts_dir)
    pred_matrix, target_columns = predict_with_ensemble(ckpt_paths, test_images, args.batch_size)
    submission = build_submission(lookup, pred_matrix, target_columns)

    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}")


if __name__ == "__main__":
    main()
