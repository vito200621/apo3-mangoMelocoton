# Guía de instalación y dependencias
# Instalación y ejecución reproducible

## Entorno

Se recomienda Python 3.11, como fija `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate fruit-sorter
```

Sin Conda, cree un entorno virtual con Python 3.11 y ejecute `python -m pip install -r requirements.txt`.

## Datos fuera de Git

1. Descargue Fruit Quality Classification desde [Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) y extraiga sus carpetas Good/Bad/Mixed bajo `data/raw/kaggle/`. No agregue imágenes originales al repositorio.
2. Organice las fotografías propias bajo `data/raw/own_collection/Good/<fruta>/`, `Regular/<fruta>/` o `Bad/<fruta>/`; aplique antes [el protocolo de anotación](protocolo_anotacion.md) y excluya información personal/EXIF.
3. La fuente pública puede no tener Regular. Ejecute el cuaderno 02 para generar plantillas Mixed, anote manualmente solo recortes identificables en `data/annotations/mixed_candidates_annotations.csv` y vuelva a ejecutar el cuaderno 02. Filas sin calidad se excluyen.

## Flujo experimental

Abra JupyterLab desde la raíz del repositorio y ejecute, en orden, los cuatro notebooks:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_feature_extraction.ipynb`
4. `notebooks/04_model_comparison.ipynb`

02 regenera las salidas procesadas; confirme que las etiquetas/plantillas están guardadas y use el reinicio solo cuando quiera reemplazar los derivados existentes. 03 guarda modelos locales en `models/` y resultados de validación en `experiments/results/training/`. 04 evalúa test después de la selección y guarda tablas/figuras SVG y PNG en `experiments/results/comparison/`. Esos artefactos no se versionan.

La alternativa de línea de comandos es `python -m src.main eda --reset-processed`, luego `python -m src.main train --skip-eda` y `python -m src.main evaluate`. Para repetir el pipeline completo use `python -m src.main all --reset-processed`. La CLI no reemplaza los análisis y figuras de los notebooks 03/04.

## Aplicación y pruebas

Con los modelos ya entrenados, ejecute `streamlit run src/app/interface.py`. La interfaz permite cargar una imagen o usar la captura de cámara; no se ha validado latencia de vídeo en tiempo real. Pruebe una imagen ilegible, formato no permitido y ausencia de los artefactos antes de la demostración.

Ejecute `python -m pytest` desde la raíz. La evaluación requiere dependencias instaladas desde `requirements.txt`; una instalación parcial o un Python distinto al soportado puede impedir TensorFlow/OpenCV.

La evaluación final requiere los datos descargados, las etiquetas revisadas y los modelos entrenados. Sin esos artefactos, el flujo no produce resultados de tres clases.
