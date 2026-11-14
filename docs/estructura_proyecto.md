# **Estructura de Repositorio**

```text
apo3-mangoMelocoton/
│
├── data/                                      # Datos del proyecto
│   ├── raw/                                   # Datos originales sin procesar
│   │   ├── collected/                         # Imágenes recolectadas por el equipo
│   │   └── fruit_quality_classification/      # Dataset público de Kaggle
│   │
│   ├── processed/                             # Datos procesados
│   │   ├── images/                            # Imágenes preprocesadas
│   │   ├── features/                          # Características extraídas
│   │   └── annotations.csv                    # Etiquetas y anotaciones
│   │
│   └── splits/                                # Particiones del conjunto de datos
│       ├── train.csv                          # Datos de entrenamiento
│       ├── validation.csv                     # Datos de validación
│       └── test.csv                           # Datos de prueba
│
├── docs/                                      # Documentación del proyecto
│   ├── project/                               # Documentos del proyecto académico
│   │   ├── 0_Proyecto_Final_26-2.md
│   │   ├── 0_Proyecto Final 26-2.pdf
│   │   ├── Rubrica.md
│   │   └── Rúbrica.pdf
│   │
│   ├── README.md                              # Documentación extendida
│   ├── arquitectura.md                        # Arquitectura del sistema
│   ├── estructura_proyecto.md                 # Organización del repositorio
│   ├── metodologia.md                         # Metodología CRISP-DM
│   ├── analisis_etico.md                      # Análisis ético del proyecto
│   └── instalacion.md                         # Instalación y configuración
│
├── src/                                       # Código fuente principal
│   ├── __init__.py
│   ├── main.py                                # CLI y coordinación del flujo
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   └── interface.py                       # Interfaz Streamlit
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                         # Manifiestos y anotaciones
│   │   ├── feature_extraction.py              # Características clásicas
│   │   ├── load_data.py                       # Imágenes y arrays preparados
│   │   ├── prepare_data.py                    # Preparación y particiones
│   │   ├── preprocess.py                      # Transformaciones y guardado de imagen
│   │   ├── segmentation.py                    # Segmentación de imágenes mixtas
│   │   └── size_estimation.py                  # Tamaño relativo
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── compare_models.py                  # Coordinación y exportación de resultados
│   │   ├── evaluate_cnn.py
│   │   ├── evaluate_random_forest.py
│   │   ├── evaluate_svm.py
│   │   └── metrics.py                         # Métricas compartidas
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── random_forest.py
│   │   └── svm.py
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_cnn.py
│   │   ├── train_models.py                    # Coordinador de entrenamiento
│   │   ├── train_random_forest.py
│   │   └── train_svm.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                          # Metadatos de etiquetas y pesos
│       └── helpers.py                         # Persistencia compartida
│
├── notebooks/                                 # Experimentación y análisis
│   ├── 01_data_exploration.ipynb              # Exploración de datos
│   ├── 02_preprocessing.ipynb                 # Análisis del preprocesamiento
│   ├── 03_feature_extraction.ipynb            # Extracción de características
│   └── 04_model_comparison.ipynb              # Comparación de modelos
│
├── experiments/                               # Seguimiento de experimentos
│   ├── logs/                                  # Registros de entrenamiento
│   │   ├── svm/
│   │   ├── random_forest/
│   │   └── cnn/
│   │
│   ├── checkpoints/                           # Modelos entrenados
│   │   ├── svm/
│   │   ├── random_forest/
│   │   └── cnn/
│   │
│   └── results/                               # Resultados experimentales
│       ├── svm/
│       ├── random_forest/
│       ├── cnn/
│       ├── comparison/                        # Comparación de modelos
│       └── figures/                            # Gráficas y visualizaciones
│
├── tests/
│   ├── __init__.py
│   ├── test_cnn.py
│   ├── test_dataset.py
│   ├── test_load_data.py
│   ├── test_predictor.py
│   ├── test_preprocessing.py
│   ├── test_random_forest.py
│   ├── test_segmentation.py
│   ├── test_size_estimation.py
│   └── test_svm.py
│
├── requirements.txt                           # Dependencias de Python
├── environment.yml                            # Configuración del entorno Conda
├── .gitignore                                 # Archivos excluidos de Git
├── LICENSE                                    # Licencia del proyecto
└── README.md                                  # Presentación principal del proyecto
```
