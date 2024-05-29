import os
import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from src.exception import NerException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {col: LabelEncoder() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col].fit(X[col])
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return self.columns if input_features is None else input_features

class DataTransformation:
    
    def __init__(self,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.preprocessor_obg_file = os.path.join('artifacts', 'preprocessor.pkl')

    def drop_columns(self, df, columns_to_drop):
        return df.drop(columns=columns_to_drop, axis=1)
    
    def drop_columns_function(self, df):
        return self.drop_columns(df, self.data_transformation_config.column_to_drop)

    def get_data_transformation_object(self):
        try:
            columns_to_drop = self.data_transformation_config.column_to_drop  # Add the columns you want to drop
            

            cat_cols = self.data_transformation_config.one_hot_encoding
    

            label_encode_cols = self.data_transformation_config.label_encoder  # Add the columns to label encode
            

            num_cols = self.data_transformation_config.numerical_column
            
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("standard_scalar", StandardScaler(with_mean=False))
            ])

            label_pipeline = Pipeline(steps=[
                ('label_encoder', CustomLabelEncoder(columns=label_encode_cols))
            ])

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")
            logging.info(f"Label Encode Columns: {label_encode_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("drop_columns", FunctionTransformer(self.drop_columns_function), columns_to_drop),
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols),
                    ("label_pipeline", label_pipeline, label_encode_cols)
                ],
                remainder='passthrough'
            )
            return preprocessor

        except Exception as e:
            raise NerException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = self.data_transformation_config.target_column

            # Divide the train dataset into independent and dependent features
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide the test dataset into independent and dependent features
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying Preprocessing on training and test dataframes")
        
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.preprocessor_obg_file,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor_obg_file
            )

        except Exception as e:
            raise NerException(e, sys)
