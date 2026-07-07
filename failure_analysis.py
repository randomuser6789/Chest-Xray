"""Surface the model's highest-confidence false positives/negatives, with images and full labels, for manual review."""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from calibrate import apply_platt, fit_platt, sigmoid
from data import IMAGE_DIR, TARGET_LABELS, ChestXrayDataset, make_splits
from evaluate import CHECKPOINT_PATH, get_device, load_model, run_inference

FAILURES_DIR = "failures"
MANIFEST_CSV = os.path.join(FAILURES_DIR, "manifest.csv")


def top_false_positives(y_true, y_prob, n):
    """Indices (into the full test array) of the n highest-probability negatives, sorted worst-first."""
    idx = np.where(y_true == 0)[0]
    order = np.argsort(-y_prob[idx])[:n]
    return idx[order]


def top_false_negatives(y_true, y_prob, n):
    """Indices (into the full test array) of the n lowest-probability positives, sorted worst-first."""
    idx = np.where(y_true == 1)[0]
    order = np.argsort(y_prob[idx])[:n]
    return idx[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--n", type=int, default=15, help="Number of worst FP and FN cases to surface per class.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _train_df, val_df, test_df = make_splits()
    print(f"Val set: {len(val_df)} images (for fitting Platt scaling)")
    print(f"Test set: {len(test_df)} images")

    val_ds = ChestXrayDataset(val_df, train=False)
    test_ds = ChestXrayDataset(test_df, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = load_model(args.checkpoint, device)

    val_labels, val_logits, _val_ages, _val_sexes = run_inference(model, val_loader, device)
    test_labels, test_logits, _test_ages, _test_sexes = run_inference(model, test_loader, device)

    # Same Platt fit as evaluate.py: fit on val logits only, apply to test logits, so thresholds
    # for "highest confidence" below are on calibrated probabilities, not raw sigmoid outputs.
    platt_a, platt_b = fit_platt(val_logits, val_labels)
    test_probs = sigmoid(apply_platt(test_logits, platt_a, platt_b))

    os.makedirs(FAILURES_DIR, exist_ok=True)
    manifest_rows = []

    for class_idx, name in enumerate(TARGET_LABELS):
        y_true = test_labels[:, class_idx]
        y_prob = test_probs[:, class_idx]

        fp_indices = top_false_positives(y_true, y_prob, args.n)
        fn_indices = top_false_negatives(y_true, y_prob, args.n)

        print(f"\n{name}: surfacing {len(fp_indices)} FP and {len(fn_indices)} FN cases")

        for error_type, indices in [("FP", fp_indices), ("FN", fn_indices)]:
            for row_idx in indices:
                row = test_df.iloc[row_idx]
                image_index = row["Image Index"]
                pred_prob = float(y_prob[row_idx])

                dst_filename = f"{name.lower()}_{error_type}_p{pred_prob:.2f}_{image_index}"
                src_path = os.path.join(IMAGE_DIR, image_index)
                dst_path = os.path.join(FAILURES_DIR, dst_filename)
                shutil.copy2(src_path, dst_path)

                manifest_rows.append({
                    "image_index": image_index,
                    "class": name,
                    "error_type": error_type,
                    "pred_prob": pred_prob,
                    "label_pneumothorax": int(row["Pneumothorax"]),
                    "label_effusion": int(row["Effusion"]),
                    "label_cardiomegaly": int(row["Cardiomegaly"]),
                    "original_finding_labels": row["Finding Labels"],
                    "saved_filename": dst_filename,
                })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved {len(manifest_df)} images -> {FAILURES_DIR}/")
    print(f"Saved manifest -> {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
