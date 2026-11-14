from __future__ import annotations

import joblib
import numpy as np

import enum
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.data.feature_extraction import extract_features
from src.data.preprocess import preprocess_for_cnn, preprocess_image
from src.data.size_estimation import estimate_size_from_array

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "models"
_PROCESSED_DIR = _ROOT / "data" / "processed"

_LABEL_INFO_CANDIDATES = [
    _PROCESSED_DIR / "label_info.json",
    _PROCESSED_DIR / "labelinfo.json",
]

_FALLBACK_CLASS_NAMES = ["Good", "Regular", "Bad"]


class ModelBackend(enum.Enum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    CNN = "cnn"


@dataclass
class PredictionResult:
    class_index: int
    class_name: str
    confidence: float
    backend: ModelBackend
    size_label: str
    diameter_norm: float | None


def _find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_label_info() -> dict:
    label_info_path = _find_first_existing(_LABEL_INFO_CANDIDATES)

    if label_info_path is None:
        return {
            "class_names": list(_FALLBACK_CLASS_NAMES),
            "label_map": {"Good": 0, "Regular": 1, "Bad": 2},
            "inverse_map": {"0": "Good", "1": "Regular", "2": "Bad"},
        }

    with open(label_info_path, encoding="utf-8") as f:
        info = json.load(f)

    if "class_names" not in info:
        raise ValueError(f"Invalid label info file: {label_info_path}")

    return info


class AgroQualityPredictor:
    def __init__(self, backend: ModelBackend = ModelBackend.RANDOM_FOREST) -> None:
        self.backend = backend

        self.label_info = _load_label_info()
        self.class_names = self.label_info["class_names"]

        if self.class_names != ["Good", "Regular", "Bad"]:
            raise ValueError(
                "Invalid class mapping. Expected ['Good', 'Regular', 'Bad'], "
                f"got {self.class_names}. Regenerate processed metadata and retrain if needed."
            )

        self._rf = None
        self._svm = None
        self._scaler = None
        self._cnn = None

        self._load_models()

    def _resolve_model_path(self, candidates: list[str]) -> Path:
        existing = _find_first_existing([_MODELS_DIR / name for name in candidates])

        if existing is None:
            raise FileNotFoundError(
                f"None of these model files were found in {_MODELS_DIR}: {candidates}"
            )

        return existing

    def _load_models(self) -> None:
        if self.backend == ModelBackend.RANDOM_FOREST:
            rf_path = self._resolve_model_path(
                [
                    "random_forest.pkl",
                    "randomforest.pkl",
                ]
            )
            self._rf = joblib.load(rf_path)

        elif self.backend == ModelBackend.SVM:
            svm_path = self._resolve_model_path(
                [
                    "svm.pkl",
                ]
            )
            scaler_path = self._resolve_model_path(
                [
                    "scaler.pkl",
                ]
            )
            self._svm = joblib.load(svm_path)
            self._scaler = joblib.load(scaler_path)

        elif self.backend == ModelBackend.CNN:
            import tensorflow as tf

            cnn_path = self._resolve_model_path(
                [
                    "cnn_best.keras",
                    "cnnbest.keras",
                ]
            )
            self._cnn = tf.keras.models.load_model(str(cnn_path))

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def predict_frame(self, bgr_frame: np.ndarray) -> Optional[PredictionResult]:
        if bgr_frame is None:
            return None

        size_estimate = estimate_size_from_array(bgr_frame)

        if self.backend in (ModelBackend.RANDOM_FOREST, ModelBackend.SVM):
            quality_result = self._predict_ml(bgr_frame)
        else:
            quality_result = self._predict_cnn(bgr_frame)

        if quality_result is None:
            return None

        quality_result.size_label = size_estimate["size_label"]
        diameter = size_estimate["diameter_norm"]
        quality_result.diameter_norm = (
            float(diameter) if np.isfinite(diameter) else None
        )
        return quality_result

    def _predict_ml(self, bgr_frame: np.ndarray) -> Optional[PredictionResult]:
        preprocessed = preprocess_image(bgr_frame)

        if preprocessed is None:
            return None

        features = extract_features(preprocessed)

        if features is None:
            return None

        features_2d = features.reshape(1, -1)

        if self.backend == ModelBackend.RANDOM_FOREST:
            pred_idx = int(self._rf.predict(features_2d)[0])

            if hasattr(self._rf, "predict_proba"):
                proba_vec = self._rf.predict_proba(features_2d)[0]
                confidence = float(np.max(proba_vec))
            else:
                confidence = 1.0

        else:
            features_scaled = self._scaler.transform(features_2d)
            pred_idx = int(self._svm.predict(features_scaled)[0])

            if hasattr(self._svm, "predict_proba"):
                proba_vec = self._svm.predict_proba(features_scaled)[0]
                confidence = float(np.max(proba_vec))
            else:
                confidence = 1.0

        return PredictionResult(
            class_index=pred_idx,
            class_name=self._safe_class_name(pred_idx),
            confidence=confidence,
            backend=self.backend,
            size_label="unknown",
            diameter_norm=None,
        )

    def _predict_cnn(self, bgr_frame: np.ndarray) -> Optional[PredictionResult]:
        batch = preprocess_for_cnn(bgr_frame)

        if batch is None:
            return None

        proba_vec = self._cnn.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(proba_vec))
        confidence = float(proba_vec[pred_idx])

        return PredictionResult(
            class_index=pred_idx,
            class_name=self._safe_class_name(pred_idx),
            confidence=confidence,
            backend=self.backend,
            size_label="unknown",
            diameter_norm=None,
        )

    def _safe_class_name(self, idx: int) -> str:
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)
