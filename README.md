![alt text](https://github.com/helenyaben/dtu_mlops_project/blob/master/dataset-cover.jpg)
# MLOPS project - Image classification of finger pictures

Helena Sofía Yaben Lopezosa - s213617 \
Ania Bzinkowska - s213027 \
Antoni  Wojciech Skrobisz - s213612 \
Oliver Alman Sande - s174032 \
David Immanuel Hartel - s212588 

## Project description

### Overall goal of the project
The goal of the project is to use deep learning to solve a classification task of counting fingers as well as distinguishing between left and right hand.

### Data
We are using the Kaggle dataset [Fingers](https://www.kaggle.com/datasets/koryakinp/fingers), which contains 21600 images of centered left and right hands fingers. All images are 128 by 128 pixels and have a noise pattern on the background.

Training set: 18000 images \
Test set: 3600 images 

### Frameworks
 - Organization and version control: Git, cookie cutter, DVC
 - Deep Learning: PyTorch
 - Reproducibility: Docker
...



### Deep learning model
We intend to use a CNN for the image classification



## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
