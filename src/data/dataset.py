from __future__ import annotations

import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

FINAL_QUALITY_CLASSES = ["Good", "Regular", "Bad"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def infer_quality_from_folder(folder_name: str) -> str | None:
    normalized = clean_text(folder_name)

    if "mixed" in normalized:
        return "MixedPending"
    if "regular" in normalized:
        return "Regular"
    if "good" in normalized:
        return "Good"
    if "bad" in normalized:
        return "Bad"

    return None


def clean_fruit_name(folder_name: str) -> str:
    fruit = str(folder_name)

    for token in [
        "_Good",
        "_Bad",
        "_Mixed",
        "_Regular",
        "Good",
        "Bad",
        "Mixed",
        "Regular",
        "Quality",
        "Fruits",
        "Fruit",
    ]:
        fruit = fruit.replace(token, "")

    fruit = fruit.replace("_", " ").replace("-", " ").strip()
    return fruit if fruit else "unknown"


def make_sample_id(source_name: str, img_path: Path, base_path: Path) -> str:
    rel_path = img_path.relative_to(base_path).with_suffix("").as_posix()
    short_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:10]

    fruit_name = re.sub(r"[^A-Za-z0-9]+", "_", img_path.parent.name).strip("_")
    fruit_name = fruit_name[:24] if fruit_name else "unknown"

    stem_name = re.sub(r"[^A-Za-z0-9]+", "_", img_path.stem).strip("_")
    stem_name = stem_name[:24] if stem_name else "img"

    source_name = re.sub(r"[^A-Za-z0-9]+", "_", str(source_name)).strip("_")
    source_name = source_name[:16] if source_name else "src"

    return f"{source_name}_{fruit_name}_{stem_name}_{short_hash}"


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def scan_labeled_dataset(
    base_path: Path, source_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_records = []
    mixed_records = []

    if not base_path.exists():
        print(f"Path not found. Skipping: {base_path}")
        return empty_manifest(), empty_manifest()

    for quality_folder in sorted(base_path.iterdir()):
        if not quality_folder.is_dir():
            continue

        quality_label = infer_quality_from_folder(quality_folder.name)

        if quality_label is None:
            print(f"Unrecognized folder skipped: {quality_folder.name}")
            continue

        image_paths = [
            img_path
            for img_path in sorted(quality_folder.rglob("*"))
            if img_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img_path in image_paths:
            sample_id = make_sample_id(source_name, img_path, base_path)
            fruit_name = clean_fruit_name(img_path.parent.name)

            record = {
                "sample_id": sample_id,
                "path": str(img_path),
                "source": source_name,
                "source_image_id": sample_id,
                "fruit": fruit_name,
                "quality": quality_label,
                "size_label": "unknown",
                "diameter_norm": np.nan,
                "defect": "",
                "group_id": sample_id,
                "notes": "",
            }

            if quality_label == "MixedPending":
                record["quality"] = ""
                mixed_records.append(record)
            else:
                labeled_records.append(record)

    return pd.DataFrame(labeled_records), pd.DataFrame(mixed_records)


def create_or_update_annotation_template(
    records_df: pd.DataFrame,
    annotation_path: Path,
    required_quality: bool = True,
) -> pd.DataFrame:
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    columns = empty_manifest().columns.tolist()

    if records_df.empty and not annotation_path.exists():
        return empty_manifest()

    template = records_df.copy()

    for col in columns:
        if col not in template.columns:
            template[col] = ""

    template = template[columns]

    if required_quality:
        template["quality"] = ""

    if not annotation_path.exists():
        template.to_csv(annotation_path, index=False)
        print(f"Annotation template created: {annotation_path}")
        return empty_manifest()

    annotations = pd.read_csv(annotation_path)

    for col in columns:
        if col not in annotations.columns:
            annotations[col] = ""

    annotations = annotations[columns].copy()

    existing_ids = set(annotations["sample_id"].astype(str))
    new_rows = template[~template["sample_id"].astype(str).isin(existing_ids)]

    if not new_rows.empty:
        annotations = pd.concat([annotations, new_rows], ignore_index=True)
        annotations.to_csv(annotation_path, index=False)
        print(f"Annotation template updated with {len(new_rows)} new rows.")

    annotations["quality"] = annotations["quality"].astype(str).str.strip()
    annotations["path"] = annotations["path"].astype(str)

    if required_quality:
        annotations = annotations[
            annotations["quality"].isin(FINAL_QUALITY_CLASSES)
        ].copy()

    annotations = annotations[
        annotations["path"].apply(lambda p: Path(p).exists())
    ].copy()
    return annotations


def apply_own_annotations(own_df: pd.DataFrame, annotation_path: Path) -> pd.DataFrame:
    if own_df.empty:
        return own_df

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    columns = empty_manifest().columns.tolist()

    template = own_df.copy()

    for col in columns:
        if col not in template.columns:
            template[col] = ""

    template = template[columns]

    if not annotation_path.exists():
        template.to_csv(annotation_path, index=False)
        print(f"Own-collection annotation template created: {annotation_path}")
        return own_df

    annotations = pd.read_csv(annotation_path)

    for col in columns:
        if col not in annotations.columns:
            annotations[col] = ""

    annotations = annotations[columns].copy()

    existing_ids = set(annotations["sample_id"].astype(str))
    new_rows = template[~template["sample_id"].astype(str).isin(existing_ids)]

    if not new_rows.empty:
        annotations = pd.concat([annotations, new_rows], ignore_index=True)
        annotations.to_csv(annotation_path, index=False)
        print(f"Own annotation file updated with {len(new_rows)} new rows.")

    annotation_cols = [
        "sample_id",
        "capture_device",
        "lighting",
        "background",
        "quality",
        "size_label",
        "diameter_norm",
        "diameter_mm",
        "maturity",
        "defect",
        "group_id",
        "notes",
    ]

    annotations = annotations[annotation_cols].copy()
    for col in annotation_cols[1:]:
        annotations[col] = annotations[col].replace(r"^\s*$", np.nan, regex=True) # type: ignore

    merged = own_df.merge(
        annotations,
        on="sample_id",
        how="left",
        suffixes=("", "_ann"),
    )

    for col in [
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
    ]:
        ann_col = f"{col}_ann"
        if ann_col in merged.columns:
            merged[col] = merged[ann_col].combine_first(merged[col])
            merged = merged.drop(columns=[ann_col])

    merged = merged[merged["quality"].isin(FINAL_QUALITY_CLASSES)].copy()
    return merged


def apply_mixed_annotations(
    candidates_df: pd.DataFrame, annotation_path: Path
) -> pd.DataFrame:
    if candidates_df.empty:
        return empty_manifest()

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    columns = empty_manifest().columns.tolist()
    template = candidates_df.copy()

    for column in columns:
        if column not in template.columns:
            template[column] = ""

    template = template[columns]

    if not annotation_path.exists():
        template.to_csv(annotation_path, index=False)
        print(f"Mixed-candidate annotation template created: {annotation_path}")
        return empty_manifest()

    annotations = pd.read_csv(annotation_path)
    if "sample_id" not in annotations.columns:
        raise ValueError(f"{annotation_path} must contain a sample_id column.")
    if annotations["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{annotation_path} contains duplicate sample_id values.")

    for column in columns:
        if column not in annotations.columns:
            annotations[column] = ""

    annotations = annotations[columns].copy()
    existing_ids = set(annotations["sample_id"].astype(str))
    new_rows = template[~template["sample_id"].astype(str).isin(existing_ids)]
    if not new_rows.empty:
        annotations = pd.concat([annotations, new_rows], ignore_index=True)
        annotations.to_csv(annotation_path, index=False)
        print(f"Mixed annotation file updated with {len(new_rows)} new rows.")

    annotation_columns = [
        "sample_id",
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
        "notes",
    ]
    annotations = annotations[annotation_columns].copy()
    for column in annotation_columns[1:]:
        annotations[column] = annotations[column].replace(
            r"^\s*$", np.nan, regex=True
        )

    merged = candidates_df.merge(
        annotations,
        on="sample_id",
        how="left",
        suffixes=("", "_ann"),
        validate="one_to_one",
    )
    for column in annotation_columns[1:]:
        annotation_column = f"{column}_ann"
        if annotation_column in merged.columns:
            merged[column] = merged[annotation_column].combine_first(merged[column])
            merged = merged.drop(columns=[annotation_column])

    merged["quality"] = merged["quality"].astype("string").str.strip()
    merged["quality"] = merged["quality"].mask(merged["quality"].eq(""), pd.NA)
    invalid_quality = merged["quality"].notna() & ~merged["quality"].isin(
        FINAL_QUALITY_CLASSES
    )
    if invalid_quality.any():
        invalid_ids = merged.loc[invalid_quality, "sample_id"].astype(str).tolist()
        raise ValueError(
            "Mixed annotations contain unsupported quality labels for "
            f"sample_id values: {invalid_ids[:10]}"
        )

    return merged[merged["quality"].isin(FINAL_QUALITY_CLASSES)].copy()
