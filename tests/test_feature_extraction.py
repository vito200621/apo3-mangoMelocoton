import numpy as np
import pandas as pd
import pytest

from src.data.feature_extraction import load_split_groups


def test_split_groups_follow_feature_order_and_validate_label_alignment(tmp_path):
    processed = tmp_path / "processed"
    records = []
    for quality, filename, group_id in (
        ("Good", "good.png", "scene-good"),
        ("Bad", "bad.png", "scene-bad"),
    ):
        folder = processed / "train" / quality
        folder.mkdir(parents=True)
        path = folder / filename
        path.touch()
        records.append(
            {
                "processed_path": str(path),
                "group_id": group_id,
                "quality": quality,
            }
        )
    pd.DataFrame(records[::-1]).to_csv(processed / "train.csv", index=False)
    label_info = {
        "class_names": ["Good", "Regular", "Bad"],
        "label_map": {"Good": 0, "Regular": 1, "Bad": 2},
    }

    groups = load_split_groups(
        processed, "train", label_info, expected_labels=np.array([0, 2])
    )

    assert groups.tolist() == ["scene-good", "scene-bad"]
    with pytest.raises(ValueError, match="do not align"):
        load_split_groups(
            processed, "train", label_info, expected_labels=np.array([2, 0])
        )
