# **Estructura de Repositorio para Entregas**

A medida que avancemos en el curso, la organización será tan importante como el código que escriban. Para garantizar que todos los proyectos sean fáciles de revisar, entender y replicar, usaremos una estructura de repositorio estandarizada para cada entrega.

## La Regla de Oro: Claridad y Reproducibilidad

Cada una de sus entregas estará contenida en su propia carpeta principal (entrega 1, entrega 2, etc.), pero dentro de ellas, siempre deben incluir la siguiente organización.

Al seguir esta estructura sugerida, nos aseguramos de que cualquier persona (incluyéndote a ti mismo en el futuro) pueda:

- Entender el Proyecto rápidamente leyendo el README.md.

- Ejecutar el Código sin problemas gracias a requirements.txt.

- Localizar el código fuente principal dentro de la carpeta src/.

## Estructura Detallada por Entrega

**IMPORTANTE**: a continuación se presenta una sugerencia detallada, no es necesario que su estructura tenga exactamente los detalles que se presentan a continuación. El objetivo es mostrar la estructura que típicamente encontraran en un proyecto.
Para cada entrega, tu carpeta debe replicar la siguiente jerarquía (hasta donde les sea posible), adaptando los archivos internos según sea necesario (esta es la estructura, pero simplificada para el contexto de una entrega y de acuerdo a sus necesidades):

```text
📦 apo3-mangoMelocoton/
│
├── 📂 data/                              # Datos del proyecto
│   ├── 📂 raw/                           # Datos originales sin procesar
│   │   ├── 📂 collected/                 # Imágenes recolectadas por el equipo
│   │   └── 📂 fruit_quality_classification/ # Dataset público de Kaggle
│   │
│   ├── 📂 processed/                     # Datos procesados
│   │   ├── 📂 images/                    # Imágenes preprocesadas
│   │   ├── 📂 features/                  # Características extraídas
│   │   └── 📜 annotations.csv            # Etiquetas y anotaciones
│   │
│   └── 📂 splits/                        # Particiones del conjunto de datos
│       ├── 📜 train.csv                  # Datos de entrenamiento
│       ├── 📜 validation.csv             # Datos de validación
│       └── 📜 test.csv                   # Datos de prueba
│
├── 📂 docs/                              # Documentación del proyecto
│   ├── 📂 project/                       # Documentos del proyecto académico
│   │   ├── 📜 0_Proyecto_Final_26-2.md
│   │   ├── 📜 0_Proyecto Final 26-2.pdf
│   │   ├── 📜 Rubrica.md
│   │   └── 📜 Rúbrica.pdf
│   │
│   ├── 📜 README.md                      # Documentación extendida
│   ├── 📜 arquitectura.md                # Arquitectura del sistema
│   ├── 📜 estructura_proyecto.md         # Organización del repositorio
│   ├── 📜 metodologia.md                 # Metodología CRISP-DM
│   ├── 📜 analisis_etico.md              # Análisis ético del proyecto
│   └── 📜 instalacion.md                 # Instalación y configuración
│
├── 📂 src/                               # Código fuente principal
│   ├── 📂 app/                           # Interfaz y despliegue
│   │   ├── 📜 interface.py               # Interfaz gráfica con Streamlit
│   │   ├── 📜 predictor.py               # Lógica de predicción
│   │   └── 📜 __init__.py
│   │
│   ├── 📂 data/                          # Procesamiento y preparación de datos
│   │   ├── 📜 load_data.py               # Carga de datos
│   │   ├── 📜 preprocess.py              # Preprocesamiento de imágenes
│   │   ├── 📜 segmentation.py            # Segmentación de frutas y verduras
│   │   ├── 📜 feature_extraction.py      # Extracción de características
│   │   ├── 📜 dataset.py                 # Preparación de conjuntos de datos
│   │   └── 📜 __init__.py
│   │
│   ├── 📂 models/                        # Definición de modelos
│   │   ├── 📜 svm.py                     # Modelo SVM
│   │   ├── 📜 xgboost_model.py           # Modelo XGBoost
│   │   ├── 📜 cnn.py                     # Red neuronal convolucional
│   │   └── 📜 __init__.py
│   │
│   ├── 📂 training/                      # Entrenamiento de modelos
│   │   ├── 📜 train_svm.py
│   │   ├── 📜 train_xgboost.py
│   │   ├── 📜 train_cnn.py
│   │   └── 📜 __init__.py
│   │
│   ├── 📂 evaluation/                    # Evaluación y comparación
│   │   ├── 📜 evaluate_svm.py
│   │   ├── 📜 evaluate_xgboost.py
│   │   ├── 📜 evaluate_cnn.py
│   │   ├── 📜 metrics.py                 # Métricas compartidas
│   │   ├── 📜 compare_models.py          # Comparación de modelos
│   │   └── 📜 __init__.py
│   │
│   ├── 📂 utils/                         # Funciones auxiliares
│   │   ├── 📜 config.py                  # Configuración general
│   │   ├── 📜 helpers.py                 # Funciones reutilizables
│   │   ├── 📜 logger.py                  # Registro de eventos
│   │   └── 📜 __init__.py
│   │
│   └── 📜 __init__.py
│
├── 📂 notebooks/                         # Experimentación y análisis
│   ├── 📜 01_data_exploration.ipynb      # Exploración de datos
│   ├── 📜 02_preprocessing.ipynb         # Análisis del preprocesamiento
│   ├── 📜 03_feature_extraction.ipynb    # Extracción de características
│   └── 📜 04_model_comparison.ipynb      # Comparación de modelos
│
├── 📂 experiments/                       # Seguimiento de experimentos
│   ├── 📂 logs/                          # Registros de entrenamiento
│   │   ├── 📂 svm/
│   │   ├── 📂 xgboost/
│   │   └── 📂 cnn/
│   │
│   ├── 📂 checkpoints/                   # Modelos entrenados
│   │   ├── 📂 svm/
│   │   ├── 📂 xgboost/
│   │   └── 📂 cnn/
│   │
│   └── 📂 results/                       # Resultados experimentales
│       ├── 📂 svm/
│       ├── 📂 xgboost/
│       ├── 📂 cnn/
│       ├── 📂 comparison/                # Comparación de modelos
│       └── 📂 figures/                   # Gráficas y visualizaciones
│
├── 📂 tests/                             # Pruebas unitarias e integración
│   ├── 📜 test_preprocessing.py
│   ├── 📜 test_segmentation.py
│   ├── 📜 test_svm.py
│   ├── 📜 test_xgboost.py
│   ├── 📜 test_cnn.py
│   ├── 📜 test_predictor.py
│   └── 📜 __init__.py
│
├── 📜 requirements.txt                   # Dependencias de Python
├── 📜 environment.yml                    # Configuración del entorno Conda
├── 📜 .gitignore                         # Archivos excluidos de Git
├── 📜 LICENSE                            # Licencia del proyecto
└── 📜 README.md                          # Presentación principal del proyecto
```
