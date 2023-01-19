![alt text](https://github.com/helenyaben/dtu_mlops_project/blob/master/final_report/figures/dataset-cover.jpg)
# MLOPS project - Image classification of finger pictures

Helena Sofía Yaben Lopezosa - s213617 \
Ania Bzinkowska - s213027 \
Antoni  Wojciech Skrobisz - s213612 \
Oliver Alman Sande - s174032 \
David Immanuel Hartel - s212588 

## Project description


### Overall goal of the project
The goal of the project is to develop MLOps pipeline for computer vision problem. We will use deep learning to solve a classification task of counting fingers as well as distinguishing between left and right hand.


### What framework are we going to use and how do we intend to include the framework into your project

 - Organization and version control: Git, cookie cutter, DVC
 - Deep Learning: PyTorch
 - Reproducibility: Docker
 - Experiment logging: Weights & Biases
 - Minimizing boilerplate
 - Continuous Integration: Unittesting, Github actions
 - Google Cloud deployment


### What data are we going to run the model on
We are using the Kaggle [Fingers](https://www.kaggle.com/datasets/koryakinp/fingers) dataset, which contains 21600 images of centered left and right hands 'showing numbers' with fingers. All images are 128 by 128 pixels and have a noise pattern on the background.

Training set: 18000 images \
Test set: 3600 images 

Labels are in the last 2 characters of a file name. L/R indicates left/right hand; 0,1,2,3,4,5 indicates the number of fingers.


### What deep learning models do we expect to use
We intend to utilise a CNN for the image classification. The CNNs will serve as a backbone of our model, and then the extracted features will be passsed to a fully connected layer to make predictions. The specificity of the task does not require more sophisticated network, which will allow us to focus on all the elements of the pipeline to be delevoped.


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, data sets for training and testing.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── report       <- Final report in markdown format
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
    ├── tests              <- Pytests for data, model and traingin and coverage report
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
