from dataclasses import dataclass
import os
from pathlib import Path
from src.constants import *
from src.entity.config_entity import *


#Data Ingestion
@dataclass
class DataIngestionConfig:

    def __init__ (self):

        self.data_ingestion_artifacts_dir:str = os.path.join(
            ARTIFACTS_DIR,ROOT_DIR
        )

        self.source_url:str = SOURCE_URL

        self.local_data_file:str = os.path.join(
            self.data_ingestion_artifacts_dir,LOCAL_FILE_PATH
        )

        self.unzip_dir:str = UNZIP_DIR

        self.unzip_csv_data:str = os.path.join(
            ARTIFACTS_DIR,UNZIP_DIR_CSV_DATA
        )

#Data Transformation
@dataclass
class DataTransformationConfig:

    def __init__ (self):

        self.data_transformation_dir:str = os.path.join(
            ARTIFACTS_DIR,DATA_TRANSFORMATION_DIR
        )
        self.data_transformation_file_name: str = DATA_TRANSFORMATION_FILE

        self.column_to_drop:str = COLUMNS_TO_DROP

        self.is_certified:str = IS_CERTIFIED
        
        self.is_confidential:str = IS_CONFIDENTIAL

        self.registration_date:str = REGISTRATION_DATE

        self.registration_year:str = REGISTRATION_YEAR
        
        self.year:str = YEAR

        self.country_column:str = COUNTRY_COLUMN

        self.countries:str = COUNTRIES

        self.column_convert_to_int:str = CONVERT_TO_INT

        self.label_encoder:str = LABEL_ENCODER

        self.target_encoding:str = TARGET_ENCODING

        self.target_column:str = TARGET_COLUMN

        self.one_hot_encoding:str = ONE_HOT_ENCODING

        self.standard_scaler:str = STANDARDSCALER

        self.model_dir:str = os.path.join(
            MODEL_DIR
        )

        self.pickle_file_name:str = TRANSFORM_PICKLE_FILE_NAME

        self.train_test_file_path:str = os.path.join(
            ARTIFACTS_DIR,TRAIN_TEST_SET_DATA_LOCATION
        )

#Model Trainer 
@dataclass
class ModelTrainerConfig:

    def __init__(self):
        
        self.model_trainer_dir:str = os.path.join(
            ARTIFACTS_DIR,MODEL_TRAINING_ARTIFACTS_DIR
        )

        self.xgboost_params = PARAM_GRID

        self.random_forest_params = PARAM_GRID_RF
        