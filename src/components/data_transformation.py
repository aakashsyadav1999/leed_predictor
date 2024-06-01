import os
import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import NerException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object

@dataclass
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

@dataclass
class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.preprocessor_obj_file = os.path.join('MODEL_DIR', 'preprocessor.pkl')

    def drop_columns(self, df, columns_to_drop):
        if not isinstance(df, pd.DataFrame):
            logging.error("drop_columns: Input should be a DataFrame")
            raise ValueError("Input should be a DataFrame")
        return df.drop(columns=columns_to_drop, axis=1)

    def drop_columns_function(self, df):
        return self.drop_columns(df, self.data_transformation_config.column_to_drop)

    def get_data_transformation_object(self):
        try:
            columns_to_drop = self.data_transformation_config.column_to_drop

            cat_cols = self.data_transformation_config.one_hot_encoding
            ordinal_encode_cols = self.data_transformation_config.label_encoder
            num_cols = self.data_transformation_config.numerical_column

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
            ])

            ordinal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
            ])

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")
            logging.info(f"Ordinal Encode Columns: {ordinal_encode_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_pipeline", cat_pipeline, cat_cols),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_encode_cols),
                    ("num_pipeline", num_pipeline, num_cols),
                    ("drop_columns", FunctionTransformer(self.drop_columns_function), columns_to_drop),
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

            # Handle missing values in the target feature
            train_df[target_column_name].fillna(train_df[target_column_name].mode()[0], inplace=True)

            test_df[target_column_name].fillna(test_df[target_column_name].mode()[0], inplace=True)

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

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
                file_path=self.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return (
            train_arr,
            test_arr,
            self.preprocessor_obj_file
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise NerException(e, sys)
