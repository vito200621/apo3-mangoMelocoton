from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.dataset import IMAGE_EXTENSIONS
from src.data.load_data import load_image
from src.data.preprocess import load_and_preprocess

FEATURE_DIM = 108
HSV_BINS = 32


def extract_features(img_bgr: np.ndarray) -> np.ndarray | None:
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    features: list[float] = []

    # HSV histograms: 32 bins x 3 channels = 96 features
    for channel in range(3):
        hist = cv2.calcHist([img_hsv], [channel], None, [HSV_BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist.tolist())

    # RGB channel statistics: normalized mean + std per channel = 6 features
    for channel in range(3):
        vals = img_rgb[:, :, channel].flatten().astype(np.float32)
        features.append(float(vals.mean() / 255.0))
        features.append(float(vals.std() / 255.0))

    # HSV channel statistics: normalized mean + std per channel = 6 features
    for channel in range(3):
        vals = img_hsv[:, :, channel].flatten().astype(np.float32)
        features.append(float(vals.mean() / 255.0))
        features.append(float(vals.std() / 255.0))

    vec = np.array(features, dtype=np.float32)

    assert vec.shape == (
        FEATURE_DIM,
    ), f"Expected {FEATURE_DIM} features, got {vec.shape[0]}"

    return vec


def extract_features_from_path(path: str | Path) -> np.ndarray | None:
    img = load_and_preprocess(path)

    if img is None:
        return None

    return extract_features(img)


def extract_features_from_processed_path(path: str | Path) -> np.ndarray | None:
    img = load_image(path)

    if img is None:
        return None

    return extract_features(img)


def build_feature_matrix(
    paths: list[str | Path], verbose: bool = False, already_preprocessed: bool = False
) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, path in enumerate(paths):
        if already_preprocessed:
            vec = extract_features_from_processed_path(path)
        else:
            vec = extract_features_from_path(path)

        if vec is not None:
            rows.append(vec)
            valid_indices.append(i)

        if verbose and (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(paths)} images")

    if not rows:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    return np.vstack(rows), valid_indices


def load_split_features(
    processed_dir: str | Path, split_name: str, label_info: dict
) -> tuple[np.ndarray, np.ndarray]:
    processed_dir = Path(processed_dir)

    X: list[np.ndarray] = []
    y: list[int] = []

    label_map = label_info["label_map"]
    class_names = label_info["class_names"]

    for quality in class_names:
        label = label_map[quality]
        folder = processed_dir / split_name / quality

        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        img_files = []

        for ext in IMAGE_EXTENSIONS:
            img_files.extend(folder.glob(f"*{ext}"))
            img_files.extend(folder.glob(f"*{ext.upper()}"))

        img_files = sorted(set(img_files))

        for img_path in tqdm(img_files, desc=f"{split_name}/{quality}"):
            vec = extract_features_from_processed_path(img_path)

            if vec is not None:
                X.append(vec)
                y.append(label)

    if not X:
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int32)
