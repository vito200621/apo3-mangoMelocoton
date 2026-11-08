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

3. Download the [Fruit Quality Classification dataset](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) and keep the raw data outside version control, or follow the team's shared-data instructions.
4. Use [notebooks/](notebooks/) for experimentation, [src/](src/) for preprocessing, training, evaluation, and application code, [experiments/](experiments/) for generated results, and [tests/](tests/) for tests.
5. Consult [docs/estructura_proyecto.md](docs/estructura_proyecto.md) for the repository organization and reproducibility guidelines.

## Featured Notebooks/Analysis/Deliverables

- [Experiment notebook](notebooks/experiment_1.ipynb)
- [Final project requirements and evaluation criteria](docs/project/0_Proyecto_Final_26-2.md)
- [Repository structure guidelines](docs/estructura_proyecto.md)
