"""Calibration curves, per-subgroup AUC (sex, age), and also failure taxonomy analysis."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import TARGET_LABELS, ChestXrayDataset, make_splits
from model import build_model

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "eval_results.csv")

MIN_POSITIVES = 10
N_CALIBRATION_BINS = 10

AGE_BUCKETS = [
    ("<40", 0, 40),
    ("40-60", 40, 60),
    ("60-80", 60, 80),
    ("80+", 80, 1000),
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    model = build_model(freeze_features=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, loader, device):
    """Run the model over a loader, returning labels, sigmoid probabilities, ages, and sexes as arrays."""
    all_labels = []
    all_probs = []
    all_ages = []
    all_sexes = []

    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_ages.append(meta["age"].numpy())
            all_sexes.extend(meta["sex"])

    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    ages = np.concatenate(all_ages, axis=0)
    sexes = np.array(all_sexes)

    return labels, probs, ages, sexes


def safe_auc(y_true, y_score, min_positives=MIN_POSITIVES):
    """AUC if the slice has at least min_positives positives and both classes present; else (None, reason)."""
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n_pos < min_positives:
        return None, f"n_pos={n_pos} < min {min_positives}"
    if n_pos == n:
        return None, "no negatives present"

    return roc_auc_score(y_true, y_score), None


def compute_ece(y_true, y_prob, n_bins=N_CALIBRATION_BINS):
    """Expected Calibration Error: bin by predicted prob, weight |observed freq - mean predicted prob| by bin size."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    bin_stats = []

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            bin_stats.append(None)
            continue
        confidence = y_prob[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (count / n) * abs(accuracy - confidence)
        bin_stats.append((confidence, accuracy, count))

    return ece, bin_stats


def plot_calibration_curve(y_true, y_prob, class_name, ece, save_path):
    _, bin_stats = compute_ece(y_true, y_prob)

    confidences = [s[0] for s in bin_stats if s is not None]
    accuracies = [s[1] for s in bin_stats if s is not None]
    counts = [s[2] for s in bin_stats if s is not None]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(confidences, accuracies, s=[max(20, c / 2) for c in counts], color="C0")
    ax.plot(confidences, accuracies, color="C0", alpha=0.5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{class_name}\nECE={ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def age_bucket_label(age):
    for label, lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return label
    return None


def evaluate_slice(labels, probs, mask, class_idx, min_positives=MIN_POSITIVES):
    y_true = labels[mask, class_idx]
    y_score = probs[mask, class_idx]
    n = int(mask.sum())
    n_pos = int(y_true.sum())
    auc, reason = safe_auc(y_true, y_score, min_positives=min_positives)
    return {"n": n, "n_pos": n_pos, "auc": auc, "note": reason or ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _train_df, _val_df, test_df = make_splits()
    print(f"Test set: {len(test_df)} images (held out, never used in training)")

    test_ds = ChestXrayDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = load_model(args.checkpoint, device)
    labels, probs, ages, sexes = run_inference(model, test_loader, device)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    print("\nOverall test AUC:")
    for i, name in enumerate(TARGET_LABELS):
        result = evaluate_slice(labels, probs, np.ones(len(labels), dtype=bool), i)
        auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else f"NA ({result['note']})"
        print(f"  {name}: AUC={auc_str} (n={result['n']}, n_pos={result['n_pos']})")
        rows.append({"subgroup_type": "overall", "subgroup_value": "all", "class": name, **result})

    print("\nCalibration (ECE) per class:")
    for i, name in enumerate(TARGET_LABELS):
        ece, _ = compute_ece(labels[:, i], probs[:, i])
        print(f"  {name}: ECE={ece:.4f}")
        save_path = os.path.join(FIGURES_DIR, f"calibration_{name.lower()}.png")
        plot_calibration_curve(labels[:, i], probs[:, i], name, ece, save_path)
        print(f"    saved plot -> {save_path}")
        for row in rows:
            if row["subgroup_type"] == "overall" and row["class"] == name:
                row["ece"] = ece

    print("\nSubgroup AUC by sex:")
    for sex_value in sorted(set(sexes)):
        mask = sexes == sex_value
        for i, name in enumerate(TARGET_LABELS):
            result = evaluate_slice(labels, probs, mask, i)
            auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else f"NA ({result['note']})"
            print(f"  sex={sex_value} {name}: AUC={auc_str} (n={result['n']}, n_pos={result['n_pos']})")
            rows.append({"subgroup_type": "sex", "subgroup_value": sex_value, "class": name, **result})

    print("\nSubgroup AUC by age bucket:")
    bucket_labels = np.array([age_bucket_label(a) for a in ages])
    for label, _lo, _hi in AGE_BUCKETS:
        mask = bucket_labels == label
        for i, name in enumerate(TARGET_LABELS):
            result = evaluate_slice(labels, probs, mask, i)
            auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else f"NA ({result['note']})"
            print(f"  age={label} {name}: AUC={auc_str} (n={result['n']}, n_pos={result['n_pos']})")
            rows.append({"subgroup_type": "age_bucket", "subgroup_value": label, "class": name, **result})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results table -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
