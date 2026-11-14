from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.evaluation.metrics import full_report
from src.utils.config import load_label_info


def build_test_generator(
    processed_dir: str | Path = Path("data/processed"),
    batch_size: int = 32,
) -> tuple[object, dict]:
    processed_dir = Path(processed_dir)
    label_info = load_label_info(processed_dir)
    generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        processed_dir / "test",
        target_size=tuple(label_info["img_size"]),
        batch_size=batch_size,
        class_mode="sparse",
        classes=label_info["class_names"],
        shuffle=False,
    )
    if generator.samples == 0:
        raise ValueError(f"No test images found under {processed_dir / 'test'}.")
    if generator.class_indices != label_info["label_map"]:
        raise ValueError("CNN test classes do not match the saved label mapping.")
    return generator, label_info


def evaluate_cnn(
    processed_dir: str | Path = Path("data/processed"),
    models_dir: str | Path = Path("models"),
    batch_size: int = 32,
) -> dict:
    models_dir = Path(models_dir)
    model_path = next(
        (
            path
            for path in (models_dir / "cnn_best.keras", models_dir / "cnnbest.keras")
            if path.exists()
        ),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(f"CNN model not found in {models_dir}.")

    generator, label_info = build_test_generator(processed_dir, batch_size)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(generator, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return full_report(
        generator.classes, predictions, probabilities,
        label_info["class_names"], "CNN",
    )
