"""Training loop for the DenseNet-121 multi-label classifier on MPS."""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import TARGET_LABELS, ChestXrayDataset, make_splits
from model import build_model

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(train_df):
    """pos_weight[c] = num_negatives[c] / num_positives[c], for BCEWithLogitsLoss on imbalanced labels."""
    labels = train_df[TARGET_LABELS].values.astype(np.float32)
    num_pos = labels.sum(axis=0)
    num_neg = len(train_df) - num_pos
    pos_weight = num_neg / num_pos

    print("Per-class positive counts in train set:")
    for name, pos, neg, w in zip(TARGET_LABELS, num_pos, num_neg, pos_weight):
        rate = pos / (pos + neg) * 100
        print(f"  {name}: {int(pos)} pos / {int(neg)} neg ({rate:.1f}% positive) -> pos_weight={w:.2f}")

    return torch.tensor(pos_weight, dtype=torch.float32)


def subset_df(df, n, seed=42):
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over loader. Trains if optimizer is given, otherwise evaluates and returns AUCs too."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels = []
    all_logits = []

    with torch.set_grad_enabled(is_train):
        for images, labels, _meta in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_labels.append(labels.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    return avg_loss, all_labels, all_logits


def per_class_auc(labels, logits):
    aucs = []
    for i, name in enumerate(TARGET_LABELS):
        y_true = labels[:, i]
        y_score = logits[:, i]
        if len(np.unique(y_true)) < 2:
            print(f"  {name}: AUC undefined (only one class present in this batch of data)")
            aucs.append(np.nan)
            continue
        auc = roc_auc_score(y_true, y_score)
        print(f"  {name}: AUC={auc:.4f}")
        aucs.append(auc)
    return aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-features", action="store_true")
    parser.add_argument("--subset", type=int, default=None,
                         help="Train/val on a small subset of N images, for a fast smoke test.")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_df, val_df, _test_df = make_splits()
    if args.subset is not None:
        train_df = subset_df(train_df, args.subset)
        val_df = subset_df(val_df, max(args.subset // 5, 20))
    print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")

    train_ds = ChestXrayDataset(train_df, train=True)
    val_ds = ChestXrayDataset(val_df, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(freeze_features=args.freeze_features).to(device)

    pos_weight = compute_pos_weight(train_df).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_macro_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        print(f"  train_loss={train_loss:.4f}")

        val_loss, val_labels, val_logits = run_epoch(model, val_loader, device, criterion, optimizer=None)
        print(f"  val_loss={val_loss:.4f}")
        print("  Val per-class AUC:")
        val_aucs = per_class_auc(val_labels, val_logits)
        macro_auc = float(np.nanmean(val_aucs))
        print(f"  val_macro_auc={macro_auc:.4f}")

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  New best macro-AUC ({macro_auc:.4f}) -> saved to {CHECKPOINT_PATH}")

    print(f"\nDone. Best val macro-AUC: {best_macro_auc:.4f}")


if __name__ == "__main__":
    main()
