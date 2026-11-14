from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.evaluation.metrics import evaluate_predictions
from src.models.cnn import build_cnn
from src.utils.config import load_class_weights, load_label_info

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")

def build_cnn_generators(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    batch_size: int = 64,
) -> tuple[object, object, object, dict]:
    processed_dir = Path(processed_dir)
    label_info = load_label_info(processed_dir)

    class_names = label_info["class_names"]
    img_size = tuple(label_info["img_size"])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.10,
        height_shift_range=0.10,
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        processed_dir / "train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        classes=class_names,
        shuffle=True,
    )

    val_gen = eval_datagen.flow_from_directory(
        processed_dir / "val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        classes=class_names,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        processed_dir / "test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        classes=class_names,
        shuffle=False,
    )

    if train_gen.class_indices != label_info["label_map"]:
        raise ValueError(
            "Keras class_indices do not match labelinfo.json. "
            "Check folder names and class order."
        )

    return train_gen, val_gen, test_gen, label_info


def train_cnn(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    epochs: int = 15,
    batch_size: int = 64,
) -> tuple[tf.keras.Model, dict]:
    tf.keras.utils.set_random_seed(42)
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, _, label_info = build_cnn_generators(
        processed_dir=processed_dir,
        batch_size=batch_size,
    )

    class_names = label_info["class_names"]
    n_classes = label_info["n_classes"]
    img_size = tuple(label_info["img_size"])
    class_weights = load_class_weights(processed_dir)

    cnn = build_cnn(
        input_shape=(img_size[0], img_size[1], 3),
        n_classes=n_classes,
    )

    best_model_path = models_dir / "cnn_best.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = cnn.fit(
        train_gen, # type: ignore
        validation_data=val_gen, # type: ignore
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    cnn = tf.keras.models.load_model(best_model_path)
    y_pred_val = np.argmax(cnn.predict(val_gen, verbose=1), axis=1)
    y_true_val = val_gen.classes # type: ignore

    metrics = evaluate_predictions(
        y_true=y_true_val,
        y_pred=y_pred_val,
        class_names=class_names,
        model_name="CNN",
        split="Validation",
    )
    metrics["history"] = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    metrics["saved_model_path"] = str(best_model_path)

    return cnn, metrics
