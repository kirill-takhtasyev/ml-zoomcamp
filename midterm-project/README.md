# ML Zoomcamp Midterm Project. Stroke Prediction Model
This folder contains the work completed as part of the Midterm project of the 2023 cohort [ML Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp), which is lead by Alexey Grigorev. The course is accessible for enrollment at your convenience and is offered free of charge.
# Problem description
A stroke, also known as a cerebrovascular accident (CVA), is a medical condition that occurs when there is a sudden disruption of blood flow to a part of the brain. This interruption in blood supply can lead to brain damage and a variety of symptoms.

According to the World Health Organization (WHO) and various other health agencies, stroke is indeed one of the leading causes of death worldwide. It is responsible for a significant number of fatalities, accounting for approximately 11% of total deaths globally. This highlights the importance of stroke awareness, prevention, and the need for timely and effective medical interventions to reduce its impact on public health.

The model is constructed using the [dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data), which was created by [fedesoriano](https://www.kaggle.com/fedesoriano). It is designed for predicting the probability of stroke based on the provided attribute information.

This is a binary classification problem.
# Project files
- `README.md` - file with project description and instructions;
- `healthcare-dataset-stroke-data.csv` - the CSV file containing the Stroke Prediction Dataset. Was downloaded from Kaggle and provided by [fedesoriano](https://www.kaggle.com/fedesoriano);
- `notebook.ipynb` - file in Jupyter Notebook format containing Exploratory Data Analysis and training process of different models;
- `train.py` - separate script, which contains the data preparation, training process of the final model and code for saving it to file;
- `model.bin` - file containing the final model;
- `predict.py` - Flask app with final function used for creating the web service;
- `predict_test.py` - example script created for checking the model, running on a server, and getting the response;
- `environment.yml` - environment file for creating a new Conda virtual environment with the packages listed in `requirements.txt`;
-
# Setting up the environment
For managing the environments i used Conda - the open-source package management system, which can be installed on Windows, macOS and Linux. Here is the [Free Download Link](https://www.anaconda.com/download).
