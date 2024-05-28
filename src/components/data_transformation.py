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
            self.encoders[col].fit(X[col].astype(str))
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col].astype(str))
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return self.columns if input_features is None else input_features

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, target=None):
        self.columns = columns
        self.target = target
        self.encoders = {col: TargetEncoder() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col].fit(X[col].astype(str), y.astype(str))
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X[col].astype(str))
        return X_copy

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)



class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.transformed_data = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder() 

    def read_csv(self):
        try:
            data_directory = self.data_transformation_config.data_transformation_dir
            files = os.listdir(data_directory)
            csv_files = [file for file in files if file.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the directory")
            csv_file_path = os.path.join(data_directory, csv_files[0])
            df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False, skipinitialspace=True)
            df.isnull().sum()
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
            logging.info(f"Data columns: {df.columns.tolist()}")
            df = df[df[self.data_transformation_config.is_certified] == 'Yes']
            df = df[df[self.data_transformation_config.is_confidential] == 'No']
            df[self.data_transformation_config.registration_year] = pd.to_datetime(df[self.data_transformation_config.registration_date])
            df[self.data_transformation_config.year] = df[self.data_transformation_config.registration_year].dt.year
            df = df[df[self.data_transformation_config.year] >= 2015]

            if self.data_transformation_config.country_column in df.columns:
                df = df[df[self.data_transformation_config.country_column].isin(self.data_transformation_config.countries)]
            else:
                raise KeyError(f"Column '{self.data_transformation_config.country_column}' not found in DataFrame")
            df.dropna(inplace=True)
            df = df.fillna(0)
            logging.info(f"Separating data {df.head()}")
            logging.info(f"Separating data {df.columns}")
            logging.info(f"Data dtypes of columns {df.dtypes}")
            logging.info(f"Data dtypes of columns {df.isnull().sum()}")
            df.to_csv("cleaned_separated_data.csv")
            
            return df
        except Exception as e:
            logging.error(f"Error separating data: {e}")
            raise

    def create_pipeline(self, label_encode_columns, one_hot_encode_columns, target_encode_columns, drop_columns):
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('label_encoder', CustomLabelEncoder(columns=label_encode_columns), label_encode_columns),
                    ('one_hot_encoder', OneHotEncoder(sparse=True), one_hot_encode_columns),
                    ('target_encoder', CustomTargetEncoder(columns=target_encode_columns, target=self.data_transformation_config.target_column), target_encode_columns),
                    ('drop_columns', DropColumnsTransformer(columns=drop_columns), drop_columns)
                ],
                remainder='passthrough'
            )

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logging.info("Preprocessing pipeline created successfully.")
            return pipeline
        except Exception as e:
            logging.error(f"Error during pipeline creation: {e}")
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

    def transform(self):
        try:
            # Step 1: Read and clean the data
            df = self.read_csv()
            logging.info(f"Data shape after reading CSV: {df.shape}")

            # Step 2: Separate the data
            df_new = self.separating_data(df)
            logging.info(f"Data shape after separating: {df_new.shape}")
            logging.info(f"Data types after separating: {df_new.dtypes}")

            # Step 3: Prepare column lists
            label_encode_columns = [col for col in self.data_transformation_config.label_encoder if col != self.data_transformation_config.target_column]
            target_encode_columns = self.data_transformation_config.target_encoding
            drop_columns = self.data_transformation_config.column_to_drop
            one_hot_encode_columns = [col for col in self.data_transformation_config.one_hot_encoding if col != self.data_transformation_config.target_column]

            # Ensure all columns to be encoded are strings
            df_new = self.ensure_column_types(df_new, label_encode_columns, 'object')
            df_new = self.ensure_column_types(df_new, one_hot_encode_columns, 'object')
            df_new = self.ensure_column_types(df_new, target_encode_columns, 'object')

            # Step 4: Encode the target column to numerical values
            target_column = self.data_transformation_config.target_column
            y = df_new[target_column]
            logging.info(f"Target column after encoding: {df_new[target_column].head()}")

            # Step 5: Separate features (X) and target (y)
            X = df_new.drop(columns=[target_column])
            logging.info(f"Feature columns: {X.columns.tolist()}")
            logging.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

            # Step 6: Create and fit the preprocessing pipeline
            pipeline = self.create_pipeline(label_encode_columns, one_hot_encode_columns, target_encode_columns, drop_columns)
            self.preprocessor = pipeline.fit(X, y)

            # Step 7: Transform the features
            X_transformed = self.preprocessor.transform(X)
            logging.info(f"Shape of transformed X: {X_transformed.shape}")

            # Step 8: Combine the transformed features back into a DataFrame
            transformed_df = pd.DataFrame(X_transformed, columns=self.preprocessor.get_feature_names_out())
            transformed_df[target_column] = y.values
            logging.info(f"Shape of transformed DataFrame: {transformed_df.shape}")

            # Step 9: Store the transformed data
            self.transformed_data = transformed_df

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
