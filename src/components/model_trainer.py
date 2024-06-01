import pandas as pd
import pickle
from dataclasses import dataclass,field
import os

from src.constants import *
from src.entity.config_entity import ModelTrainerConfig,DataTransformationConfig
from src.logger import logging

from imblearn.over_sampling import SMOTE

#Defining models for prediction
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils.common import save_object


@dataclass
#class for initiating all methods
class ModelTrainer:

    def __init__ (self,model_trainer_config:ModelTrainerConfig,data_transformation_config:DataTransformationConfig):
    
        self.model_trainer_config = model_trainer_config
        self.data_transformation_config = data_transformation_config
        self.trained_model_file_path=os.path.join('MODEL_DIR','model.pkl')

        # preprocessor_path = os.path.join('MODEL_DIR', 'preprocessor.pkl')
        # if os.path.exists(preprocessor_path):
        #     with open(preprocessor_path, 'rb') as f:
        #         self.preprocessor = pickle.load(f)

    def model_train(self,train_array,test_array):

        try:
            logging.info("Enter into Model Building")
            # Define features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Apply SMOTE to the training data for CertLevel prediction
            smote = SMOTE(k_neighbors=1, n_jobs=-1, sampling_strategy='auto', random_state=42)
            X_train,y_train = smote.fit_resample(X_train, y_train)

            # Create RandomForestClassifier
            rf_model_cert = RandomForestClassifier(random_state=42)


            # Perform GridSearchCV with increased regularization
            grid_search_rf = GridSearchCV(estimator=rf_model_cert, param_grid=self.model_trainer_config.random_forest_params, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search_rf.fit(X_train, y_train)

            # Best parameters found for CertLevel prediction with increased regularization
            print("Best Parameters for CertLevel (RandomForest with Increased Classification):", grid_search_rf.best_params_)

            # Best estimator for CertLevel prediction with increased regularization
            best_estimator_rf = grid_search_rf.best_estimator_

            # Evaluate the best estimator for CertLevel prediction with increased regularization on the test set
            accuracy_cert_rf = best_estimator_rf.score(X_test, y_test)
            print("Accuracy for CertLevel (RandomForest with Increased Classification):", accuracy_cert_rf)

            # # Evaluate the RandomForestClassifier model with increased regularization on the validation set
            # accuracy_cert_rf_val = best_estimator_rf.score(X_val, y_val)
            # print("Accuracy for CertLevel (RandomForest with Increased Regularization - Validation):", accuracy_cert_rf_val)

            # Creating directories if they don't exist
            os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)
            
            save_object(
                file_path=self.trained_model_file_path,
                obj=best_estimator_rf
            )
        
        
        except Exception as e:
            raise e


    def initiate_model_trainer(self,train_arr,test_arr):

        logging.info("Entered the initiate_model_trainer method of the model trainer class")
        try:
            os.makedirs(
                self.model_trainer_config.model_trainer_dir,exist_ok=True
            )

            self.model_train(train_arr,test_arr)

        except Exception as e:
            raise e