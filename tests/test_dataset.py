import numpy as np
import pandas as pd

import pytest

from src.data.dataset import (
    apply_mixed_annotations,
    apply_own_annotations,
    empty_manifest,
)


def test_blank_annotations_keep_folder_labels(tmp_path):
    own = empty_manifest()
    own.loc[0] = {
        "sample_id": "sample-1",
        "path": "fruit.png",
        "source": "own_collection",
        "source_image_id": "sample-1",
        "fruit": "apple",
        "quality": "Good",
        "size_label": "unknown",
        "diameter_norm": np.nan,
        "defect": "",
        "group_id": "sample-1",
        "notes": "",
    }
    annotations = pd.DataFrame([{"sample_id": "sample-1", "quality": ""}])
    path = tmp_path / "annotations.csv"
    annotations.to_csv(path, index=False)

    result = apply_own_annotations(own, path)

    assert len(result) == 1
    assert result.iloc[0]["quality"] == "Good"


def test_mixed_annotations_keep_scene_groups_and_exclude_pending_rows(tmp_path):
    candidates = empty_manifest()
    candidates.loc[0] = {
        "sample_id": "crop-1",
        "path": str(tmp_path / "crop-1.png"),
        "source": "kaggle_mixed_segmented",
        "source_image_id": "scene-1",
        "fruit": "unknown",
        "quality": "",
        "size_label": "unknown",
        "diameter_norm": 0.4,
        "defect": "",
        "group_id": "scene-1",
        "notes": "manual_quality_annotation_required",
    }
    candidates.loc[1] = {
        **candidates.iloc[0].to_dict(),
        "sample_id": "crop-2",
    }
    annotation_path = tmp_path / "mixed_annotations.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "crop-1",
                "quality": "Regular",
                "fruit": "mango",
                "size_label": "medium",
            },
            {"sample_id": "crop-2", "quality": ""},
        ]
    ).to_csv(annotation_path, index=False)

    result = apply_mixed_annotations(candidates, annotation_path)

    assert result["sample_id"].tolist() == ["crop-1"]
    assert result.iloc[0]["quality"] == "Regular"
    assert result.iloc[0]["fruit"] == "mango"
    assert result.iloc[0]["group_id"] == "scene-1"
    assert result.iloc[0]["path"] == str(tmp_path / "crop-1.png")


def test_mixed_annotations_reject_duplicate_sample_ids(tmp_path):
    candidates = empty_manifest()
    candidates.loc[0] = {
        "sample_id": "crop-1",
        "path": "crop-1.png",
        "source": "kaggle_mixed_segmented",
        "source_image_id": "scene-1",
        "fruit": "unknown",
        "quality": "",
        "size_label": "unknown",
        "diameter_norm": np.nan,
        "defect": "",
        "group_id": "scene-1",
        "notes": "",
    }
    annotation_path = tmp_path / "mixed_annotations.csv"
    pd.DataFrame(
        [{"sample_id": "crop-1"}, {"sample_id": "crop-1"}]
    ).to_csv(annotation_path, index=False)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        apply_mixed_annotations(candidates, annotation_path)
