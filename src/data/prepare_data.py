from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import (
    FINAL_QUALITY_CLASSES,
    apply_mixed_annotations,
    apply_own_annotations,
    empty_manifest,
    scan_labeled_dataset,
)
from src.data.feature_extraction import load_split_features
from src.data.preprocess import IMG_SIZE, preprocess_dataframe_to_directory
from src.data.segmentation import segment_mixed_dataset
from src.data.size_estimation import add_size_estimations

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_KAGGLE = DATA_DIR / "raw" / "kaggle"
RAW_OWN = DATA_DIR / "raw" / "own_collection"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
def ensure_dirs() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _has_generated_processed_outputs() -> bool:
    """Return whether processed data contains outputs beyond tracked placeholders."""
    if not PROCESSED_DIR.exists():
        return False

    preserved_files = {"annotations.csv", ".gitkeep"}
    placeholder_dirs = {"images", "features"}
    for path in PROCESSED_DIR.iterdir():
        if path.is_file() and path.name in preserved_files:
            continue
        if path.is_dir() and path.name in placeholder_dirs:
            if any(child.name != ".gitkeep" for child in path.iterdir()):
                return True
            continue
        return True

    return False


def _clear_generated_processed_outputs() -> None:
    """Clear generated outputs while preserving tracked placeholders and annotations."""
    if not PROCESSED_DIR.exists():
        return

    for path in list(PROCESSED_DIR.iterdir()):
        if path.is_file() and path.name in {"annotations.csv", ".gitkeep"}:
            continue
        if path.is_dir() and path.name in {"images", "features"}:
            for child in list(path.iterdir()):
                if child.is_file() and child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_eda_pipeline(reset_processed: bool = False) -> dict:
    ensure_dirs()

    print("=" * 80)
    print("STEP 1/4 - EDA + DATA PREPARATION")
    print("=" * 80)

    if not reset_processed and _has_generated_processed_outputs():
        raise FileExistsError(
            f"Processed data already exists at {PROCESSED_DIR}. "
            "Pass --reset-processed to replace it."
        )

    df_kaggle, mixed_df = scan_labeled_dataset(RAW_KAGGLE, "kaggle")
    df_own_raw, _ = scan_labeled_dataset(RAW_OWN, "own_collection")

    if df_kaggle.empty and df_own_raw.empty and mixed_df.empty:
        raise ValueError(
            "No images were found in data/raw/kaggle, data/raw/own_collection, "
            "or the Mixed folders under data/raw/kaggle."
        )

    if reset_processed:
        _clear_generated_processed_outputs()

    own_annotation_path = ANNOTATIONS_DIR / "own_collection_annotations.csv"
    df_own = apply_own_annotations(df_own_raw, own_annotation_path)

    mixed_manifest = empty_manifest()
    if not mixed_df.empty:
        mixed_output = PROCESSED_DIR / "images" / "mixed_candidates"
        mixed_manifest = segment_mixed_dataset(
            mixed_df,
            mixed_output,
            reset_output=True,
        )
        mixed_manifest_path = mixed_output / "manifest.csv"
        mixed_manifest.to_csv(mixed_manifest_path, index=False)
        mixed_annotations_path = ANNOTATIONS_DIR / "mixed_candidates_annotations.csv"
        mixed_manifest = apply_mixed_annotations(
            mixed_manifest,
            mixed_annotations_path,
        )

    manifest_parts = [df_kaggle, df_own, mixed_manifest]
    manifest_parts = [
        part for part in manifest_parts if part is not None and not part.empty
    ]

    if not manifest_parts:
        raise ValueError(
            "No annotated Good, Regular, or Bad images were found. "
            "Complete the annotation CSV templates before training."
        )

    df = pd.concat(manifest_parts, ignore_index=True)

    required_columns = [
        "sample_id",
        "path",
        "source",
        "source_image_id",
        "fruit",
        "capture_device",
        "lighting",
        "background",
        "quality",
        "maturity",
        "size_label",
        "diameter_norm",
        "diameter_mm",
        "defect",
        "group_id",
        "notes",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df = df[df["quality"].isin(FINAL_QUALITY_CLASSES)].copy()
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    if df["sample_id"].astype(str).duplicated().any():
        raise ValueError("The final manifest contains duplicate sample_id values.")

    if df.empty:
        raise ValueError("Final manifest is empty after filtering final classes.")

    print(f"Final manifest size: {len(df):,}")
    print("Class distribution:")
    print(df["quality"].value_counts().reindex(FINAL_QUALITY_CLASSES, fill_value=0))

    df = add_size_estimations(df, path_col="path", overwrite=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        for quality in FINAL_QUALITY_CLASSES:
            (PROCESSED_DIR / split / quality).mkdir(parents=True, exist_ok=True)

    class_counts = (
        df["quality"].value_counts().reindex(FINAL_QUALITY_CLASSES, fill_value=0)
    )
    available_classes = class_counts[class_counts > 0].index.tolist()

    if len(available_classes) < 2:
        raise ValueError("At least two quality classes are required to build splits.")

    use_group_split = df["group_id"].duplicated().any()

    if use_group_split:
        print("Using group-aware split.")
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(splitter1.split(df, groups=df["group_id"]))

        df_train = df.iloc[train_idx].copy()
        df_temp = df.iloc[temp_idx].copy()

        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(splitter2.split(df_temp, groups=df_temp["group_id"]))

        df_val = df_temp.iloc[val_idx].copy()
        df_test = df_temp.iloc[test_idx].copy()
    else:
        print("Using stratified split.")
        df_train, df_temp = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df["quality"],
        )
        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.50,
            random_state=42,
            stratify=df_temp["quality"],
        )

    split_frames = {"train": df_train, "validation": df_val, "test": df_test}
    for quality in available_classes:
        missing_splits = [
            split_name
            for split_name, split_df in split_frames.items()
            if not split_df["quality"].eq(quality).any()
        ]
        if missing_splits:
            raise ValueError(
                f"Class {quality!r} is missing from {missing_splits}. "
                "Add more independent groups or revise the group split."
            )

    train_counts = df_train["quality"].value_counts()
    insufficient_cv_classes = {
        quality: int(count)
        for quality, count in train_counts.items()
        if count < 3
    }
    if insufficient_cv_classes:
        raise ValueError(
            "Three-fold stratified cross-validation needs at least 3 training "
            f"examples per class; found {insufficient_cv_classes}."
        )

    train_groups = set(df_train["group_id"].astype(str))
    for split_name, split_df in (("validation", df_val), ("test", df_test)):
        overlap = train_groups & set(split_df["group_id"].astype(str))
        if overlap:
            raise ValueError(
                f"Group leakage between train and {split_name}: "
                f"{len(overlap)} shared group_id values."
            )
    if set(df_val["group_id"].astype(str)) & set(df_test["group_id"].astype(str)):
        raise ValueError("Group leakage between validation and test.")

    print(f"Train: {len(df_train):,}")
    print(f"Val:   {len(df_val):,}")
    print(f"Test:  {len(df_test):,}")

    df_train = preprocess_dataframe_to_directory(
        df_train, PROCESSED_DIR, "train", size=IMG_SIZE
    )
    df_val = preprocess_dataframe_to_directory(
        df_val, PROCESSED_DIR, "val", size=IMG_SIZE
    )
    df_test = preprocess_dataframe_to_directory(
        df_test, PROCESSED_DIR, "test", size=IMG_SIZE
    )

    split_frames = {"train": df_train, "validation": df_val, "test": df_test}
    for quality in available_classes:
        missing_splits = [
            split_name
            for split_name, split_df in split_frames.items()
            if not split_df["quality"].eq(quality).any()
        ]
        if missing_splits:
            raise ValueError(
                f"Class {quality!r} has no readable processed images in "
                f"{missing_splits}."
            )

    postprocess_train_counts = df_train["quality"].value_counts()
    insufficient_cv_classes = {
        quality: int(count)
        for quality, count in postprocess_train_counts.items()
        if count < 3
    }
    if insufficient_cv_classes:
        raise ValueError(
            "Three-fold cross-validation needs at least 3 readable training "
            f"images per class; found {insufficient_cv_classes}."
        )

    train_groups = set(df_train["group_id"].astype(str))
    for split_name, split_df in (("validation", df_val), ("test", df_test)):
        if train_groups & set(split_df["group_id"].astype(str)):
            raise ValueError(f"Group leakage after preprocessing in {split_name}.")
    if set(df_val["group_id"].astype(str)) & set(df_test["group_id"].astype(str)):
        raise ValueError("Group leakage after preprocessing between validation and test.")

    split_summary = {
        "seed": 42,
        "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "strategy": "group-aware" if use_group_split else "stratified",
        "splits": {
            split_name: {
                "samples": int(len(split_df)),
                "groups": int(split_df["group_id"].nunique()),
                "class_counts": {
                    quality: int(split_df["quality"].eq(quality).sum())
                    for quality in FINAL_QUALITY_CLASSES
                },
            }
            for split_name, split_df in split_frames.items()
        },
        "shared_groups": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }

    label_map = {quality: idx for idx, quality in enumerate(FINAL_QUALITY_CLASSES)}
    inverse_map = {str(idx): quality for quality, idx in label_map.items()}

    label_info = {
        "label_map": label_map,
        "inverse_map": inverse_map,
        "n_classes": len(label_map),
        "class_names": FINAL_QUALITY_CLASSES,
        "img_size": list(IMG_SIZE),
    }

    with open(PROCESSED_DIR / "labelinfo.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2)

    for subset in [df_train, df_val, df_test]:
        subset["label"] = subset["quality"].map(label_map)

    present_labels = np.array(sorted(df_train["label"].dropna().astype(int).unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_labels,
        y=df_train["label"].astype(int).values, # type: ignore
    )
    class_weights = {int(k): float(v) for k, v in zip(present_labels, weights)}

    with open(PROCESSED_DIR / "classweights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)
    with open(PROCESSED_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    df.to_csv(PROCESSED_DIR / "final_manifest.csv", index=False)
    df_train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Extracting 108-dim ML features...")
    X_train, y_train = load_split_features(PROCESSED_DIR, "train", label_info)
    X_val, y_val = load_split_features(PROCESSED_DIR, "val", label_info)
    X_test, y_test = load_split_features(PROCESSED_DIR, "test", label_info)

    np.save(PROCESSED_DIR / "Xtrainml.npy", X_train)
    np.save(PROCESSED_DIR / "ytrainml.npy", y_train)
    np.save(PROCESSED_DIR / "Xvalml.npy", X_val)
    np.save(PROCESSED_DIR / "yvalml.npy", y_val)
    np.save(PROCESSED_DIR / "Xtestml.npy", X_test)
    np.save(PROCESSED_DIR / "ytestml.npy", y_test)

    print("EDA/preparation completed.")
    return {
        "manifest_size": len(df),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "split_summary": split_summary,
    }



