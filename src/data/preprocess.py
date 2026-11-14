import cv2
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image as PILImage
from src.data.load_data import load_image

IMG_SIZE: tuple[int, int] = (128, 128)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(
    img_bgr: np.ndarray, size: tuple[int, int] = IMG_SIZE
) -> np.ndarray | None:
    if img_bgr is None:
        return None
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels.")
    if len(size) != 2 or any(dimension <= 0 for dimension in size):
        raise ValueError("Image size must contain two positive dimensions.")

    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    enhanced = apply_clahe(resized)
    return enhanced


def preprocess_for_cnn(
    img_bgr: np.ndarray, size: tuple[int, int] = IMG_SIZE
) -> np.ndarray | None:
    processed = preprocess_image(img_bgr, size)

    if processed is None:
        return None

    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def load_and_preprocess(
    path: str | Path, size: tuple[int, int] = IMG_SIZE
) -> np.ndarray | None:
    img = load_image(path)

    if img is None:
        return None

    return preprocess_image(img, size)


def save_preprocessed_image(
    src_path: str | Path, dest_path: str | Path, size: tuple[int, int] = IMG_SIZE
) -> bool:
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    img = load_and_preprocess(src_path, size)

    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img_rgb).save(str(dest_path))

    return True


def preprocess_dataframe_to_directory(
    dataframe: pd.DataFrame,
    processed_dir: str | Path,
    split_name: str,
    size: tuple[int, int] = IMG_SIZE,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)

    processed_rows = []
    errors = 0
    used_dest_paths = set()

    for row_idx, (_, row) in enumerate(
        tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"Processing {split_name}",
        )
    ):
        quality = str(row["quality"])

        # Stable short hash from original path.
        path_hash = hashlib.sha1(str(row["path"]).encode("utf-8")).hexdigest()[:12]

        # Keep sample_id for traceability, but add row index and path hash
        # to avoid overwriting files when sample_id is repeated.
        safe_sample_id = str(row["sample_id"]).replace("/", "_").replace("\\", "_")

        filename = f"{row_idx:06d}_{safe_sample_id}_{path_hash}.png"
        dest_path = processed_dir / split_name / quality / filename

        if str(dest_path) in used_dest_paths:
            errors += 1
            print(f"Duplicated destination path detected: {dest_path}")
            continue

        used_dest_paths.add(str(dest_path))

        try:
            ok = save_preprocessed_image(
                src_path=row["path"],
                dest_path=dest_path,
                size=size,
            )

            if not ok:
                errors += 1
                continue

            out_row = row.copy()
            out_row["processed_path"] = str(dest_path)
            processed_rows.append(out_row)

        except (OSError, ValueError, cv2.error) as exc:
            errors += 1
            print(f"Error processing {row['path']}: {exc}")

    print(f"{split_name}: {len(processed_rows):,} saved, {errors} errors")
    return pd.DataFrame(processed_rows)
