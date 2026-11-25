import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.data.load_data import load_image


def estimate_size_from_array(img_bgr: np.ndarray) -> dict:
    if img_bgr is None or img_bgr.size == 0:
        return {
            "diameter_norm": np.nan,
            "size_label": "unknown",
        }

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels.")

    h, w = img_bgr.shape[:2]
    if not h or not w:
        return {"diameter_norm": np.nan, "size_label": "unknown"}
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Basic foreground mask for fruit-like regions.
    mask = ((saturation > 30) & (value > 25) & (value < 250)).astype("uint8") * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "diameter_norm": np.nan,
            "size_label": "unknown",
        }

    largest = max(contours, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(largest)

    # Cálculo del diámetro normalizado basado en las dimensiones de la matriz ingresada
    diameter_norm = max(bw, bh) / max(h, w)

    if diameter_norm < 0.35:
        size_label = "small"
    elif diameter_norm < 0.65:
        size_label = "medium"
    else:
        size_label = "large"

    return {
        "diameter_norm": round(float(diameter_norm), 4),
        "size_label": size_label,
    }


def estimate_size_from_image(path: str | Path) -> dict:
    img_bgr = load_image(path)
    return estimate_size_from_array(img_bgr) # type: ignore


def add_size_estimations(
    dataframe: pd.DataFrame,
    path_col: str = "path",
    overwrite: bool = False,
) -> pd.DataFrame:
    df = dataframe.copy()

    if "diameter_norm" not in df.columns:
        df["diameter_norm"] = np.nan

    if "size_label" not in df.columns:
        df["size_label"] = "unknown"

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating size"):
        has_manual_diameter = pd.notna(row.get("diameter_norm"))
        has_manual_label = str(row.get("size_label", "unknown")).strip().lower() in {
            "small",
            "medium",
            "large",
        }

        if not overwrite and has_manual_diameter and has_manual_label:
            continue

        result = estimate_size_from_image(row[path_col])

        if overwrite or not has_manual_diameter:
            df.at[idx, "diameter_norm"] = result["diameter_norm"]
        if overwrite or not has_manual_label:
            df.at[idx, "size_label"] = result["size_label"]

    return df
