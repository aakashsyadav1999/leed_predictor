import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import stat
from src.entity.config_entity import DataTransformationConfig
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.transformed_data = None
        self.preprocessor = None

    def read_csv(self):
        try:
            data_directory = self.data_transformation_config.data_transformation_dir
            files = os.listdir(data_directory)
            csv_files = [file for file in files if file.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the directory")
            csv_file_path = os.path.join(data_directory, csv_files[0])
            df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False)
            return df
        except Exception as e:
            logging.error(f"No CSV file found: {e}")
            raise

    def ensure_column_types(self, df, columns, target_type):
        for col in columns:
            if df[col].dtype != target_type:
                df[col] = df[col].astype(target_type)
        return df

    def separating_data(self, df):
        try:
            df = df[df[self.data_transformation_config.is_certified] == 'Yes']
            df = df[df[self.data_transformation_config.is_confidential] == 'No']
            df[self.data_transformation_config.registration_year] = pd.to_datetime(df[self.data_transformation_config.registration_date])
            df[self.data_transformation_config.year] = df[self.data_transformation_config.registration_year].dt.year
            df = df[df[self.data_transformation_config.year] >= 2015]
            df = df[df[self.data_transformation_config.country_column].isin(self.data_transformation_config.countries)]
            df.drop(self.data_transformation_config.column_to_drop, inplace=True, axis=1)
            df = df.dropna()
            df = df.fillna(0)
            df = df.drop_duplicates()
            logging.info(f"Separating data {df.head()}")
            df.to_csv("cleaned_seprated_data.csv")
            return df
        except Exception as e:
            logging.error(f"Error separating data: {e}")
            raise

    def target_encode(self, df):
        try:
            encoder = TargetEncoder()
            target_column = self.data_transformation_config.target_column

            if isinstance(target_column, list):
                if len(target_column) == 1:
                    target_column = target_column[0]
                else:
                    raise ValueError(f"Multiple target columns specified: {target_column}")

            if target_column not in df.columns:
                raise ValueError(f"Target column {target_column} is not in DataFrame columns")

            if df[target_column].dtype == 'object':
                df[target_column] = df[target_column].astype('category').cat.codes

            y = df[target_column].values
            logging.info(f"Target column {target_column} type: {df[target_column].dtype}")
            logging.info(f"Shape of target column {target_column}: {y.shape}")

            if y.ndim != 1:
                raise ValueError(f"Target column {target_column} is not 1D after extraction. Shape: {y.shape}")

            for column in self.data_transformation_config.target_encoding:
                if df[column].dtype == 'object':
                    df[column] = df[column].astype('category').cat.codes
                df[column] = encoder.fit_transform(df[column], y)

            return df
        except Exception as e:
            logging.error(f"Error during target encoding: {e}")
            raise e

    def create_pipeline(self, df):
        try:
            label_encode_columns = self.data_transformation_config.label_encoder
            one_hot_encode_columns = self.data_transformation_config.one_hot_encoding

            logging.info(f"Label encoding columns: {label_encode_columns}")
            logging.info(f"One hot encoding columns: {one_hot_encode_columns}")

            # Remove the target column from the features to be transformed
            label_encode_columns = [col for col in label_encode_columns if col != self.data_transformation_config.target_column]
            logging.info(f"This is label_encode_columns {label_encode_columns}")
            one_hot_encode_columns = [col for col in one_hot_encode_columns if col != self.data_transformation_config.target_column]
            logging.info(f"This is one_hot_encode_column {one_hot_encode_columns}")
            df = self.ensure_column_types(df, label_encode_columns, 'object')
            logging.info(f"This data is before calling columnTransformer{df}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('label_encoder', CustomLabelEncoder(columns=label_encode_columns), label_encode_columns),
                    ('one_hot_encoder', OneHotEncoder(sparse_output=False), one_hot_encode_columns)
                ],
                remainder='passthrough'
            )

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logging.info("Preprocessing pipeline created successfully.")
            return pipeline
        except Exception as e:
            logging.error(f"Error during pipeline creation: {e}")
            raise e

    def one_hot_encoding(self, df):
        try:
            encoded_columns = pd.get_dummies(df[self.data_transformation_config.one_hot_encoding], dtype=int)
            df_encoded = pd.concat([df, encoded_columns], axis=1)
            df_encoded.drop(self.data_transformation_config.one_hot_encoding, axis=1, inplace=True)
            os.makedirs(self.data_transformation_config.train_test_file_path, exist_ok=True)
            df_encoded.to_csv(os.path.join(self.data_transformation_config.train_test_file_path, 'final_data.csv'), index=False)
            logging.info(f"After one hot encoding:\n{df_encoded.head()}")
            return df_encoded
        except Exception as e:
            logging.error(f"Error during one-hot encoding: {e}")
            raise e

    def split_train_test_split(self, df):
        try:
            target_column = self.data_transformation_config.target_column
            if target_column not in df.columns:
                raise ValueError(f"Target column {target_column} is not in DataFrame columns")
            
            y = df[target_column].values
            logging.info(f"Target column shape: {y.shape}")

            if y.ndim != 1:
                raise ValueError(f"Target column {target_column} is not 1D. Shape: {y.shape}")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Shape of train data: {train.shape}")
            logging.info(f"Shape of test data: {test.shape}")

            os.makedirs(self.data_transformation_config.train_test_file_path, exist_ok=True)
            train.to_csv(os.path.join(self.data_transformation_config.train_test_file_path, 'train.csv'), index=False)
            test.to_csv(os.path.join(self.data_transformation_config.train_test_file_path, 'test.csv'), index=False)
            logging.info("Data split into train and test data.")

            return train, test
        except Exception as e:
            logging.error(f"Error during train-test split: {e}")
            raise e

    def debug_transformation(self, df, transformer, columns):
        try:
            transformed = transformer.fit_transform(df[columns])
            logging.info(f"Transformed shape: {transformed.shape}")
            return transformed
        except Exception as e:
            logging.error(f"Error during debug transformation: {e}")
            raise e

    def transform(self):
        try:
            df = self.read_csv()
            logging.info(f"Data shape after reading CSV: {df.shape}")

            df = self.separating_data(df)
            logging.info(f"Data shape after separating: {df.shape}")

            logging.info("Applying target encoding.")
            df_transformed = self.target_encode(df)
            logging.info(f"Data shape after target encoding: {df_transformed.shape}")

            logging.info("Creating preprocessing pipeline.")
            preprocessor = self.create_pipeline(df_transformed)

            logging.info("Applying preprocessing pipeline.")
            features = df_transformed.drop(columns=[self.data_transformation_config.target_column])
            transformed_features = preprocessor.fit_transform(features)
            logging.info(f"Data shape after preprocessing pipeline: {transformed_features.shape}")

            df_transformed = pd.DataFrame(transformed_features, columns=preprocessor.get_feature_names_out())
            df_transformed[self.data_transformation_config.target_column] = df[self.data_transformation_config.target_column].values
            logging.info(f"Data shape after converting to DataFrame: {df_transformed.shape}")

            self.transformed_data = df_transformed
            self.preprocessor = preprocessor
            logging.info("Data transformed successfully.")
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise e

    def save_preprocessor(self):
        try:
            os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)
            preprocessor_file_path = os.path.join(self.data_transformation_config.model_dir, 'preprocessor.pkl')
            with open(preprocessor_file_path, 'wb') as pickle_file:
                pickle.dump(self.preprocessor, pickle_file)
            logging.info("Preprocessor saved successfully.")
        except Exception as e:
            logging.error(f"Error saving preprocessor: {e}")
            raise e

    def initiate_data_transformation(self):
        logging.info("Entered the initiate_data_transformation method of the data ingestion class")
        try:
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            os.chmod(self.data_transformation_config.data_transformation_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"Creating {self.data_transformation_config.data_transformation_dir} directory")

            self.transform()
            self.save_preprocessor()
            logging.info("Train and test arrays saved as a single pickle file.")
        except Exception as e:
            logging.error(f"Error during data transformation initiation: {e}")
            raise e

def ensure_column_types(df, columns, target_type):
    for col in columns:
        if df[col].dtype != target_type:
            df[col] = df[col].astype(target_type)
    return df
