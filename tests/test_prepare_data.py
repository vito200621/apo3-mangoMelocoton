import json

import cv2
import numpy as np
import pytest

from src.data import prepare_data


def test_reset_does_not_clear_processed_data_when_sources_are_missing(
    tmp_path, monkeypatch
):
    raw_kaggle = tmp_path / "raw_kaggle"
    raw_own = tmp_path / "raw_own"
    processed = tmp_path / "processed"
    annotations = tmp_path / "annotations"
    processed.mkdir()
    sentinel = processed / "existing-output.csv"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(prepare_data, "RAW_KAGGLE", raw_kaggle)
    monkeypatch.setattr(prepare_data, "RAW_OWN", raw_own)
    monkeypatch.setattr(prepare_data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(prepare_data, "ANNOTATIONS_DIR", annotations)

    with pytest.raises(ValueError, match="No images were found"):
        prepare_data.run_eda_pipeline(reset_processed=True)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_pipeline_saves_stratified_three_class_split_summary(tmp_path, monkeypatch):
    raw_kaggle = tmp_path / "raw_kaggle"
    raw_own = tmp_path / "raw_own"
    processed = tmp_path / "processed"
    annotations = tmp_path / "annotations"
    for quality, color in (
        ("Good", (0, 180, 0)),
        ("Regular", (0, 150, 180)),
        ("Bad", (0, 0, 180)),
    ):
        folder = raw_kaggle / quality / "Apple"
        folder.mkdir(parents=True)
        for index in range(6):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(image, (32, 32), 12 + index, color, -1)
            assert cv2.imwrite(str(folder / f"apple_{index:02}.png"), image)

    monkeypatch.setattr(prepare_data, "RAW_KAGGLE", raw_kaggle)
    monkeypatch.setattr(prepare_data, "RAW_OWN", raw_own)
    monkeypatch.setattr(prepare_data, "PROCESSED_DIR", processed)
    monkeypatch.setattr(prepare_data, "ANNOTATIONS_DIR", annotations)

    result = prepare_data.run_eda_pipeline(reset_processed=True)

    with (processed / "split_summary.json").open(encoding="utf-8") as file:
        split_summary = json.load(file)
    assert result["manifest_size"] == 18
    assert split_summary["strategy"] == "stratified"
    assert split_summary["shared_groups"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    for split_name in ("train", "validation", "test"):
        assert all(
            split_summary["splits"][split_name]["class_counts"][quality] > 0
            for quality in ("Good", "Regular", "Bad")
        )
