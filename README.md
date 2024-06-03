# Leed Predictor

## Overview
The LEED Predictor project is designed to forecast the likelihood of buildings achieving LEED (Leadership in Energy and Environmental Design) certification using machine learning techniques. LEED certification is a globally recognized standard for green building and sustainability, and predicting this certification can help architects, engineers,
and sustainability consultants make informed decisions during the design and construction phases.

## File Descriptions

    Key Features of the LEED Predictor Repository
    LEED Certification Prediction: Utilizes machine learning algorithms to predict the likelihood of a building achieving LEED certification.
    Data Preprocessing: Includes comprehensive data cleaning and preprocessing steps to prepare data for modeling.
    Model Training and Evaluation: Provides scripts for training various machine learning models and evaluating their performance.
    Feature Engineering: Implements feature engineering techniques to enhance the predictive power of the models.
    Visualization Tools: Offers tools for visualizing data distributions, correlations, and model performance metrics.
    User-Friendly Interface: May include a simple interface or scripts to make predictions on new data easily.
    Short Description
    This repository is designed to predict the likelihood of a building achieving LEED (Leadership in Energy and Environmental Design) certification using machine learning techniques. It includes data preprocessing, feature engineering, model training, and evaluation tools.

    Long Description
    The LEED Predictor repository is a comprehensive tool aimed at predicting the potential of buildings to achieve LEED certification, which is a globally recognized symbol of sustainability achievement. The repository leverages various machine learning algorithms to analyze building data and provide accurate predictions.

    Key components of the repository include:

    Data Preprocessing: Scripts for cleaning and preparing the raw data for analysis, ensuring it is in a suitable format for model training.
    Feature Engineering: Techniques to extract and create meaningful features from the data, enhancing the model's predictive capabilities.
    Model Training and Evaluation: Tools to train different machine learning models, such as regression, classification, and ensemble methods, and evaluate their performance using metrics like accuracy, precision, recall, and F1-score.
    Visualization Tools: Functions to visualize data characteristics and model performance, aiding in better understanding and interpretation of results.
    Prediction Interface: A user-friendly interface or script that allows users to input new building data and receive LEED certification predictions.
    This repository serves as a valuable resource for developers, data scientists, and sustainability professionals looking to leverage machine learning for sustainable building certification.

### Source Code Files
- **src/components/data_ingestion.py**: [Fetching data from Source]
- **src/components/data_transformation.py**: [Transforming data using pipeline approach to make pickle for flawless data_transformation]
- **src/components/model_trainer.py**: [Building machine learning model]
- **src/constants/__init__.py**: [All constant mention for model and data transforming]
- **src/entity/config_entity.py**: [Creating callable classes]
- **src/logger/__init__.py**: [For creating log files]
- **src/exception/__init__.py**: [Details about errors]
- **src/pipeline/prediction.py**: [Making prediction]
- **src/pipeline/training.py**: [Initiating model training]

### Top-Level Files
- **end_point.py**: [END-POINT]
- **app.py**: [TESTING_DATA]

### Research Files
- **research/leed_predictor_one.py**: [Exploring dataset]
- ...

## Getting Started
 Follow below step for installation and running file on your localhost:

1. **Clone the repository:**
   ```sh
   git clone <repository-url>

2. **Navigate to the project directory::**
    ```sh
    cd <project-directory>

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt

4. **Run the application:**
    ```sh
    python main.py

5. **Run the application:**
    ```sh
    docker pull aakashsyadav1999/leedpredictor