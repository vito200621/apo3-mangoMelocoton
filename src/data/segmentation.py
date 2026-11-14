from __future__ import annotations

import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image as PILImage
from src.data.load_data import load_image


def segment_fruits_from_image(
    img_path: str | Path,
    output_dir: str | Path,
    source_image_id: str,
    min_area_ratio: float = 0.01,
    padding: int = 10,
) -> list[dict]:
    img_path = Path(img_path)
    output_dir = Path(output_dir)

    img_bgr = load_image(img_path)

    if img_bgr is None:
        return []

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 30) & (value > 25) & (value < 250)).astype("uint8") * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * h * w

    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        if (x2 - x1) > 20 and (y2 - y1) > 20:
            boxes.append((x1, y1, x2, y2, area))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    if not boxes:
        boxes = [(0, 0, w, h, h * w)]

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_records = []

    for idx, (x1, y1, x2, y2, area) in enumerate(boxes, start=1):
        crop = img_bgr[y1:y2, x1:x2]
        crop_name = f"{source_image_id}_crop_{idx:02d}.png"
        crop_path = output_dir / crop_name

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        PILImage.fromarray(crop_rgb).save(str(crop_path))

        diameter_norm = max(x2 - x1, y2 - y1) / max(h, w)

        crop_records.append(
            {
                "sample_id": f"kaggle_mixed_{source_image_id}_crop_{idx:02d}",
                "path": str(crop_path),
                "source": "kaggle_mixed_segmented",
                "source_image_id": source_image_id,
                "fruit": "unknown",
                "quality": "",
                "size_label": "unknown",
                "diameter_norm": round(float(diameter_norm), 4),
                "defect": "",
                "group_id": source_image_id,
                "notes": "manual_quality_annotation_required",
            }
        )

    return crop_records


def segment_mixed_dataset(
    mixed_df: pd.DataFrame,
    output_dir: str | Path,
    reset_output: bool = False,
) -> pd.DataFrame:
    output_dir = Path(output_dir)

    if reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Segmentation output already exists at {output_dir}. "
            "Pass reset_output=True to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    crop_records = []

    for _, row in tqdm(
        mixed_df.iterrows(),
        total=len(mixed_df),
        desc="Segmenting mixed images",
    ):
        crop_records.extend(
            segment_fruits_from_image(
                img_path=row["path"],
                output_dir=output_dir,
                source_image_id=row["source_image_id"],
            )
        )

    return pd.DataFrame(crop_records)
