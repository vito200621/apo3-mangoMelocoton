# Fruit and Vegetable Quality and Size Sorter

This project is part of the **Proyecto Final de Algoritmos y Programación III** course at Universidad Icesi, Cali, Colombia. It applies computer vision and machine learning to classify the quality and estimate the size of individual fruits and vegetables.

**Project Status:** Active

## Contributing Members

| Name | Github |
| --------- | ----------------- |
| [Alejandro Vargas Sanchez](https://github.com/alevargasschz) | @alevargasschz |
| [Juan Jose Muñoz Franco](https://github.com/JJJuuuaaannnMMMuuunnnooozzz) | @JJJuuuaaannnMMMuuunnnooozzz |
| [Victoria Restrepo B](https://github.com/vito200621) | @vito200621 |

## Objective

Given a photograph of one fruit or vegetable on a simple, uniform background, the system will:

- Classify its quality into at least three categories defined by the team.
- Estimate its size as small, medium, or large, or calculate a normalized diameter.

The project follows the **CRISP-DM** methodology and will compare at least two traditional machine learning models with a small convolutional neural network. The final predictions will be presented through a **Streamlit** interface that supports image uploads and, when available, camera capture.

### Methods Used

- Exploratory data analysis
- Image preprocessing and segmentation
- Inferential statistics
- Machine learning with hyperparameter tuning
- Convolutional neural networks
- Model evaluation and error analysis
- Data visualization
- Predictive modeling

### Technologies

- Python 3.11
- Pandas and NumPy
- OpenCV
- Scikit-learn
- TensorFlow
- Matplotlib and Seaborn
- Streamlit
- JupyterLab

## Project Description

The project addresses the slow, subjective, and error-prone manual classification of fresh products in markets and agroindustries. The main reference source is the [Fruit Quality Classification dataset on Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification), including the `mixed quality` images when individual fruits need to be segmented. The team will complement this source with 30 to 50 photographs of real fruits or vegetables collected in markets, supermarkets, or homes, covering different maturity levels, sizes, and visible defects.

The additional images will be manually annotated with quality and size labels. The analysis will examine class distributions, image quality, variability, and possible imbalance. The modeling work will compare traditional machine learning approaches and a small CNN, using predefined metrics, cross-validation or grid search for hyperparameters, a baseline, and an error analysis on the test set. The project requirements and evaluation criteria are documented in [docs/project/0_Proyecto_Final_26-2.md](docs/project/0_Proyecto_Final_26-2.md).

## Getting Started

Instructions for contributors

1. Clone this repository (for help, see the [GitHub cloning tutorial](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).
2. Create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate fruit-sorter
   ```

   Alternatively, install the dependencies listed in [requirements.txt](requirements.txt) in a Python 3.11 environment.

3. Download the [Fruit Quality Classification dataset](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) into `data/raw/kaggle` and keep raw data outside version control. The public source currently has Good/Bad and Mixed scenes, but no Regular examples. Add own images under `data/raw/own_collection/<Good|Regular|Bad>/<fruit>/`.
4. Read and agree on the [annotation protocol](docs/protocolo_anotacion.md) before labeling. The pipeline creates a Mixed-crop annotation template; only reviewed Good/Regular/Bad rows are included in training. Results from two classes are reported as a binary pilot, not as a final three-class evaluation.
5. Run notebooks [01](notebooks/01_data_exploration.ipynb), [02](notebooks/02_preprocessing.ipynb), [03](notebooks/03_feature_extraction.ipynb), and [04](notebooks/04_model_comparison.ipynb) in order. Notebooks 03 and 04 require actual data and trained artifacts; no project-trained results are committed.
6. Use [src/](src/) for preprocessing, training, evaluation, and application code, [experiments/](experiments/) for generated results, and [tests/](tests/) for tests. Consult [docs/estructura_proyecto.md](docs/estructura_proyecto.md) for reproducibility guidance.

## Running the application

From the repository root, launch the Streamlit interface with:

```bash
streamlit run src/app/interface.py
```

The command-line pipeline is available as a module:

```bash
python -m src.main --help
```

Use `--reset-processed` with `eda`, `train`, or `all` only when you want to replace the existing processed dataset.

## Notebooks and deliverables

- [Data exploration](notebooks/01_data_exploration.ipynb)
- [Preprocessing and split validation](notebooks/02_preprocessing.ipynb)
- [Feature extraction and training](notebooks/03_feature_extraction.ipynb)
- [Test evaluation and error analysis](notebooks/04_model_comparison.ipynb)
- [Annotation protocol draft](docs/protocolo_anotacion.md)
- [Final project requirements and evaluation criteria](docs/project/0_Proyecto_Final_26-2.md)
- [Repository structure guidelines](docs/estructura_proyecto.md)
