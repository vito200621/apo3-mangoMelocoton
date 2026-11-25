# Arquitectura y despliegue

## Flujo de datos

1. `src.data.dataset` inspecciona carpetas etiquetadas y mantiene identificadores de imagen/escena. Las anotaciones de Mixed se vinculan por `sample_id`, y su `group_id` se conserva desde la escena original.
2. `src.data.segmentation` genera recortes candidatos. No asigna calidad: solo candidatos revisados manualmente con Good/Regular/Bad pasan al manifiesto supervisado.
3. `src.data.prepare_data` crea particiones 70/15/15 por grupos cuando hay escenas con múltiples productos; sin grupos repetidos aplica estratificación. Comprueba soporte mínimo, fuga entre train/validation/test, preprocesa a 128 × 128 y genera arrays de 108 características.
4. `src.training` entrena Random Forest, SVM y CNN; RF/SVM hacen búsqueda Macro-F1 en validación cruzada estratificada por grupos.
5. `src.evaluation` calcula informes y conserva matrices/métricas. `notebooks/04_model_comparison.ipynb` añade gráficos y análisis de errores.
6. `src.prediction.AgroQualityPredictor` carga un backend y `src/app/interface.py` permite cargar una imagen o capturarla desde cámara con Streamlit.

## Artefactos y orden de clases

`data/processed/labelinfo.json` es la fuente de verdad para `class_names`, `label_map`, `inverse_map` y tamaño de imagen. El predictor debe usar este mapa; nunca se debe deducir el orden de clases alfabéticamente en la interfaz. Los modelos y resultados se generan localmente en `models/` y `experiments/results/`, excluidos de Git.

La CNN recibe imágenes RGB normalizadas a `[0, 1]`; RF/SVM reciben los atributos HSV/RGB guardados en los arrays `*ml.npy`. SVM requiere su `scaler.pkl` junto con `svm.pkl`. Los resultados de la app son predicciones sobre una imagen estática; no se ha medido latencia de flujo de cámara y, por tanto, no debe describirse como operación en tiempo real sin medirla.

## Ejecución y límites del despliegue

Desde la raíz, ejecute `streamlit run src/app/interface.py`. La aplicación debe mostrar errores explícitos si falta un modelo o la imagen no puede decodificarse. Antes de usarla se requiere descargar datos, completar las etiquetas necesarias, ejecutar 02 y 03, y verificar la evaluación de 04. En el checkout sin esos archivos no existen modelos finales para una demostración real.

La clasificación de tres clases solo es válida cuando Good, Regular y Bad están representadas en train, validation y test. El estimador de tamaño actual entrega una relación relativa de píxeles dependiente del encuadre; no proporciona un diámetro físico ni una clasificación validada por medida de referencia.
