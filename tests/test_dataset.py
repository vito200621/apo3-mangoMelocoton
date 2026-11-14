import numpy as np
import pandas as pd

from src.data.dataset import apply_own_annotations, empty_manifest


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
