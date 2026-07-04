"""Dataset class: loads ChestX-ray14 images and labels for Pneumothorax/Effusion/Cardiomegaly."""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

DATA_ROOT = "archive (1)"
IMAGE_DIR = os.path.join(DATA_ROOT, "images-224", "images-224")
LABEL_CSV = os.path.join(DATA_ROOT, "Data_Entry_2017.csv")
TRAIN_VAL_LIST = os.path.join(DATA_ROOT, "train_val_list_NIH.txt")
TEST_LIST = os.path.join(DATA_ROOT, "test_list_NIH.txt")

TARGET_LABELS = ["Pneumothorax", "Effusion", "Cardiomegaly"]

MAX_PLAUSIBLE_AGE = 100
VAL_FRACTION = 0.1
SPLIT_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _read_image_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_labels_df():
    """Read Data_Entry_2017.csv, drop the stray trailing column, and attach multi-hot labels + parsed age/sex."""
    df = pd.read_csv(LABEL_CSV)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    suffix = df["Patient Age"].str[-1]
    magnitude = df["Patient Age"].str[:-1].astype(int)
    age_years = magnitude.where(suffix == "Y", np.nan)
    age_years = age_years.mask(suffix == "M", magnitude / 12.0)
    age_years = age_years.mask(suffix == "D", magnitude / 365.0)
    df["Age"] = age_years

    num_sub_year = (suffix != "Y").sum()
    print(f"Converted {num_sub_year} sub-year age records (M/D suffix) to fractional years")

    df["Sex"] = df["Patient Gender"]
    df = df[df["Age"] <= MAX_PLAUSIBLE_AGE].reset_index(drop=True)

    finding_lists = df["Finding Labels"].str.split("|")
    for label in TARGET_LABELS:
        df[label] = finding_lists.apply(lambda findings, label=label: int(label in findings))

    return df


def make_splits():
    """Apply the official train_val/test image split, carving a patient-disjoint val set out of train_val."""
    df = load_labels_df()

    train_val_images = set(_read_image_list(TRAIN_VAL_LIST))
    test_images = set(_read_image_list(TEST_LIST))

    train_val_df = df[df["Image Index"].isin(train_val_images)].reset_index(drop=True)
    test_df = df[df["Image Index"].isin(test_images)].reset_index(drop=True)

    train_val_patients = train_val_df["Patient ID"].unique()
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=VAL_FRACTION, random_state=SPLIT_SEED
    )

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)].reset_index(drop=True)

    return train_df, val_df, test_df


def build_transform(train):
    """Eval transform: grayscale->3ch, ToTensor, ImageNet normalize. Train transform adds light augmentation."""
    ops = [transforms.Grayscale(num_output_channels=3)]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class ChestXrayDataset(Dataset):
    """PyTorch Dataset over ChestX-ray14 PNGs with multi-hot labels and age/sex metadata."""

    def __init__(self, df, image_dir=IMAGE_DIR, train=False, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or build_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        label = np.array([row[name] for name in TARGET_LABELS], dtype=np.float32)
        meta = {"age": float(row["Age"]), "sex": row["Sex"]}

        return image, label, meta
