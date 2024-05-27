import pandas as pd
import pickle
from dataclasses import dataclass
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

@dataclass
#class for initiating all methods
class ModelTrainer:

    def __init__ (self,model_trainer_config:ModelTrainerConfig,data_transformation_config:DataTransformationConfig):
    
        self.model_trainer_config = model_trainer_config
        self.data_transformation_config = data_transformation_config


    def read_csv_files_from_directory(self,directory):
        try:
            # List all files in the directory
            files = os.listdir(directory)

            # Filter for CSV files
            csv_files = [file for file in files if file.endswith('.csv')]

            # Check if there are any CSV files
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the directory")

            # Read the train.csv and test.csv files if they exist
            train_file_path = None
            test_file_path = None

            for file in csv_files:
                if 'train' in file.lower():
                    train_file_path = os.path.join(directory, file)
                elif 'test' in file.lower():
                    test_file_path = os.path.join(directory, file)

            if train_file_path is None or test_file_path is None:
                raise FileNotFoundError("Train or test CSV file not found in the directory")

            train_df = pd.read_csv(train_file_path, encoding='latin1', low_memory=False)
            test_df = pd.read_csv(test_file_path, encoding='latin1', low_memory=False)

            return train_df, test_df

        except Exception as e:
            logging.error(f"Error reading CSV files: {e}")
            raise

    def model_train(self,train):

        # Define features and target variables
        X = train.drop([self.data_transformation_config.column_convert_to_int, self.data_transformation_config.target_column], axis=1)
        y_points = train[self.data_transformation_config.column_convert_to_int]
        y_cert = train[self.data_transformation_config.target_column]

        # Split data into training and testing sets
        X_train, X_test, y_points_train, y_points_test, y_cert_train, y_cert_test = train_test_split(
            X, y_points, y_cert, test_size=0.2, random_state=42
            )

        # Scaling numerical features
        scaler = StandardScaler()
        columns_to_scale = [self.data_transformation_config.standard_scaler]
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # Apply SMOTE to the training data for CertLevel prediction
        smote = SMOTE(k_neighbors=1, n_jobs=-1, sampling_strategy='auto', random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_cert_train)

        # Create XGBoost classifier with increased regularization parameters
        xgb_model_cert = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

        # Perform grid search with cross-validation
        grid_search_cert = GridSearchCV(estimator=xgb_model_cert, param_grid=self.model_trainer_config.xgboost_params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_cert.fit(X_train_res, y_train_res)

        # XGBoost Regressor for PointsAchieved prediction
        xgb_model_points = xgb.XGBRegressor()
        xgb_model_points.fit(X_train, y_points_train)

        # Predict and evaluate PointsAchieved
        points_predictions = xgb_model_points.predict(X_test)
        mse = mean_squared_error(y_points_test, points_predictions)
        print(f'Mean Squared Error (PointsAchieved): {mse}')

        # Create RandomForestClassifier
        rf_model_cert = RandomForestClassifier(random_state=42)

        # Split the data into training and validation sets (80% training, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y_cert, test_size=0.2, random_state=42)


        # Perform GridSearchCV with increased regularization
        grid_search_rf = GridSearchCV(estimator=rf_model_cert, param_grid=self.model_trainer_config.random_forest_params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_rf.fit(X_train_res, y_train_res)

        # Best parameters found for CertLevel prediction with increased regularization
        #print("Best Parameters for CertLevel (RandomForest with Increased Regularization):", grid_search_rf.best_params_)

        # Best estimator for CertLevel prediction with increased regularization
        best_estimator_rf = grid_search_rf.best_estimator_

        # Evaluate the best estimator for CertLevel prediction with increased regularization on the test set
        accuracy_cert_rf = best_estimator_rf.score(X_test, y_cert_test)
        print("Accuracy for CertLevel (RandomForest with Increased Regularization):", accuracy_cert_rf)

        # Evaluate the RandomForestClassifier model with increased regularization on the validation set
        accuracy_cert_rf_val = best_estimator_rf.score(X_val, y_val)
        print("Accuracy for CertLevel (RandomForest with Increased Regularization - Validation):", accuracy_cert_rf_val)

        # Save the models and other components to a pickle file
        models_and_scalers = {
            "xgb_model_cert": grid_search_cert.best_estimator_,
            "xgb_model_points": xgb_model_points,
            "rf_model_cert": best_estimator_rf,
            "scaler": scaler
        }

        # Creating directories if they don't exist
        os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)
        
        pickle_file_path = os.path.join(self.data_transformation_config.model_dir, "models_and_scalers.pkl")
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(models_and_scalers, pickle_file)


    def initiate_model_trainer(self):

        logging.info("Entered the initiate_model_trainer method of the model trainer class")
        try:
            os.makedirs(
                self.model_trainer_config.model_trainer_dir,exist_ok=True
            )
            
            
            train_df,test_df = self.read_csv_files_from_directory(self.data_transformation_config.train_test_file_path)


            self.model_train(train_df)

        except Exception as e:
            raise e