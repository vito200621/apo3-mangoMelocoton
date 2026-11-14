from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from src.prediction.predictor import AgroQualityPredictor, ModelBackend


@st.cache_resource
def load_predictor(backend: str) -> AgroQualityPredictor:
    return AgroQualityPredictor(backend=ModelBackend(backend))


def main() -> None:
    st.set_page_config(page_title="Clasificador de frutas y verduras")
    st.title("Calidad y tamaño de frutas y verduras")
    st.write("Carga una imagen de un producto individual para obtener una predicción.")

    backend = st.selectbox(
        "Modelo de calidad",
        options=[item.value for item in ModelBackend],
        format_func=lambda value: {
            "random_forest": "Random Forest",
            "svm": "SVM",
            "cnn": "CNN",
        }[value],
    )
    uploaded = st.file_uploader(
        "Imagen del producto",
        type=["jpg", "jpeg", "png", "webp"],
    )
    captured = st.camera_input("O toma una foto")
    image_file = uploaded or captured

    if image_file is not None:
        caption = uploaded.name if uploaded is not None else "Foto capturada"
        st.image(image_file.getvalue(), caption=caption, width="stretch")

    if not st.button("Analizar imagen", disabled=image_file is None):
        return

    image = cv2.imdecode(
        np.frombuffer(image_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        st.error("No se pudo leer la imagen. Prueba con otro archivo.")
        return

    try:
        with st.spinner("Analizando la imagen..."):
            result = load_predictor(backend).predict_frame(image)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"No se pudo realizar la predicción: {error}")
        return

    if result is None:
        st.error("No fue posible generar una predicción para esta imagen.")
        return

    st.subheader("Resultado")
    st.metric("Calidad", result.class_name)
    st.metric("Confianza", f"{result.confidence:.1%}")
    st.metric("Tamaño relativo", result.size_label)
    if result.diameter_norm is not None:
        st.caption(f"Diámetro normalizado: {result.diameter_norm:.4f}")


if __name__ == "__main__":
    main()
