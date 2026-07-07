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

from calibrate import apply_platt, apply_temperature, fit_platt, fit_temperature, sigmoid
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
    """Run the model over a loader, returning labels, raw logits, ages, and sexes as arrays."""
    all_labels = []
    all_logits = []
    all_ages = []
    all_sexes = []

    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device)
            logits = model(images)

            all_labels.append(labels.numpy())
            all_logits.append(logits.cpu().numpy())
            all_ages.append(meta["age"].numpy())
            all_sexes.extend(meta["sex"])

    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    ages = np.concatenate(all_ages, axis=0)
    sexes = np.array(all_sexes)

    return labels, logits, ages, sexes


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


def safe_ece(y_true, y_prob, min_positives=MIN_POSITIVES):
    """Same thin-cell gate as safe_auc: an ECE estimate on a handful of positives is noise, not signal."""
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n_pos < min_positives:
        return None, f"n_pos={n_pos} < min {min_positives}"
    if n_pos == n:
        return None, "no negatives present"

    ece, _ = compute_ece(y_true, y_prob)
    return ece, None


def plot_calibration_curve(y_true, y_prob, class_name, ece, save_path, title_suffix=""):
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
    ax.set_title(f"{class_name}{title_suffix}\nECE={ece:.4f}")
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


def evaluate_slice(labels, probs_raw, probs_temp, probs_platt, mask, class_idx, min_positives=MIN_POSITIVES):
    """AUC (raw probs; identical under scaling) plus raw/temp/Platt ECE for one class on one subgroup slice."""
    y_true = labels[mask, class_idx]
    n = int(mask.sum())
    n_pos = int(y_true.sum())

    auc, note = safe_auc(y_true, probs_raw[mask, class_idx], min_positives=min_positives)

    if note is None:
        ece_raw, _ = safe_ece(y_true, probs_raw[mask, class_idx], min_positives=min_positives)
        ece_temp, _ = safe_ece(y_true, probs_temp[mask, class_idx], min_positives=min_positives)
        ece_platt, _ = safe_ece(y_true, probs_platt[mask, class_idx], min_positives=min_positives)
    else:
        ece_raw = None
        ece_temp = None
        ece_platt = None

    return {
        "n": n, "n_pos": n_pos, "auc": auc,
        "ece_raw": ece_raw, "ece_temp": ece_temp, "ece_platt": ece_platt,
        "note": note or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _train_df, val_df, test_df = make_splits()
    print(f"Val set: {len(val_df)} images (for fitting temperature/Platt scaling)")
    print(f"Test set: {len(test_df)} images (held out, never used in training or calibration fitting)")

    val_ds = ChestXrayDataset(val_df, train=False)
    test_ds = ChestXrayDataset(test_df, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = load_model(args.checkpoint, device)

    val_labels, val_logits, _val_ages, _val_sexes = run_inference(model, val_loader, device)
    test_labels, test_logits, test_ages, test_sexes = run_inference(model, test_loader, device)

    test_probs_raw = sigmoid(test_logits)

    print("\nFitting per-class temperature on validation logits...")
    temperature = fit_temperature(val_logits, val_labels)
    for name, t in zip(TARGET_LABELS, temperature):
        print(f"  {name}: T={t:.4f}")

    print("\nFitting per-class Platt scaling (a * logit + b) on validation logits...")
    platt_a, platt_b = fit_platt(val_logits, val_labels)
    for name, a, b in zip(TARGET_LABELS, platt_a, platt_b):
        print(f"  {name}: a={a:.4f}, b={b:.4f}")
        if a <= 0:
            print(f"    WARNING: a<=0 for {name} -- Platt scaling is NOT monotonic here, AUC will change!")

    test_logits_temp = apply_temperature(test_logits, temperature)
    test_probs_temp = sigmoid(test_logits_temp)

    test_logits_platt = apply_platt(test_logits, platt_a, platt_b)
    test_probs_platt = sigmoid(test_logits_platt)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    print("\nOverall test AUC (raw vs. temp vs. Platt -- should all match, since both scalings are monotonic):")
    for i, name in enumerate(TARGET_LABELS):
        raw_auc, _ = safe_auc(test_labels[:, i], test_probs_raw[:, i])
        temp_auc, _ = safe_auc(test_labels[:, i], test_probs_temp[:, i])
        platt_auc, _ = safe_auc(test_labels[:, i], test_probs_platt[:, i])
        if raw_auc is not None:
            assert np.isclose(raw_auc, temp_auc), f"{name}: AUC changed under temperature scaling!"
            if platt_a[i] > 0:
                assert np.isclose(raw_auc, platt_auc), f"{name}: AUC changed under Platt scaling!"
        print(f"  {name}: raw={raw_auc}, temp={temp_auc}, platt={platt_auc}")

        all_mask = np.ones(len(test_labels), dtype=bool)
        result = evaluate_slice(test_labels, test_probs_raw, test_probs_temp, test_probs_platt, all_mask, i)
        rows.append({"subgroup_type": "overall", "subgroup_value": "all", "class": name, **result})

    print("\nCalibration (ECE) per class -- raw vs. temperature vs. Platt:")
    for i, name in enumerate(TARGET_LABELS):
        ece_raw, _ = safe_ece(test_labels[:, i], test_probs_raw[:, i])
        ece_temp, _ = safe_ece(test_labels[:, i], test_probs_temp[:, i])
        ece_platt, _ = safe_ece(test_labels[:, i], test_probs_platt[:, i])
        print(f"  {name}: ECE raw={ece_raw:.4f} | temp={ece_temp:.4f} | platt={ece_platt:.4f}")

        raw_path = os.path.join(FIGURES_DIR, f"calibration_{name.lower()}.png")
        temp_path = os.path.join(FIGURES_DIR, f"calibration_{name.lower()}_temp.png")
        platt_path = os.path.join(FIGURES_DIR, f"calibration_{name.lower()}_platt.png")
        plot_calibration_curve(test_labels[:, i], test_probs_raw[:, i], name, ece_raw, raw_path)
        plot_calibration_curve(test_labels[:, i], test_probs_temp[:, i], name, ece_temp, temp_path,
                                title_suffix=" (temperature scaled)")
        plot_calibration_curve(test_labels[:, i], test_probs_platt[:, i], name, ece_platt, platt_path,
                                title_suffix=" (Platt scaled)")
        print(f"    saved plots -> {raw_path}, {temp_path}, {platt_path}")

    print("\nSubgroup AUC + ECE by sex:")
    for sex_value in sorted(set(test_sexes)):
        mask = test_sexes == sex_value
        for i, name in enumerate(TARGET_LABELS):
            result = evaluate_slice(test_labels, test_probs_raw, test_probs_temp, test_probs_platt, mask, i)
            auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else f"NA ({result['note']})"
            if result["ece_raw"] is not None:
                ece_str = f"{result['ece_raw']:.4f}|{result['ece_temp']:.4f}|{result['ece_platt']:.4f}"
            else:
                ece_str = "NA"
            print(f"  sex={sex_value} {name}: AUC={auc_str}, ECE(raw|temp|platt)={ece_str} "
                  f"(n={result['n']}, n_pos={result['n_pos']})")
            rows.append({"subgroup_type": "sex", "subgroup_value": sex_value, "class": name, **result})

    print("\nSubgroup AUC + ECE by age bucket:")
    bucket_labels = np.array([age_bucket_label(a) for a in test_ages])
    for label, _lo, _hi in AGE_BUCKETS:
        mask = bucket_labels == label
        for i, name in enumerate(TARGET_LABELS):
            result = evaluate_slice(test_labels, test_probs_raw, test_probs_temp, test_probs_platt, mask, i)
            auc_str = f"{result['auc']:.4f}" if result["auc"] is not None else f"NA ({result['note']})"
            if result["ece_raw"] is not None:
                ece_str = f"{result['ece_raw']:.4f}|{result['ece_temp']:.4f}|{result['ece_platt']:.4f}"
            else:
                ece_str = "NA"
            print(f"  age={label} {name}: AUC={auc_str}, ECE(raw|temp|platt)={ece_str} "
                  f"(n={result['n']}, n_pos={result['n_pos']})")
            rows.append({"subgroup_type": "age_bucket", "subgroup_value": label, "class": name, **result})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results table -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
