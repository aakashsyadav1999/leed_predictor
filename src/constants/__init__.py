import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts")
LOGS_DIR = "logs"
LOGS_FILE_NAME = "SIDFC.log"

#Data Ingestion
ROOT_DIR = "DataIngestionArtifacts"
SOURCE_URL= "https://drive.google.com/file/d/1uOUkKCYcRvdNIv4NagKRdT6YJ7W-EuB_/view?usp=sharing"
UNZIP_DIR= ROOT_DIR
LOCAL_FILE_PATH = "public_leed_data.zip"
UNZIP_DIR_CSV_DATA = "DataIngestionArtifacts"
SPLIT_TRAIN_TES_DATA = "DataTransformationArtifacts"


#Data Transformation
DATA_TRANSFORMATION_DIR = "DataTransformationArtifacts"
DATA_TRANSFORMATION_FILE = 'cleaned_data.csv'
COLUMNS_TO_DROP = [
                    'CertDate',
                    'RegistrationDate',
                    'State'
                ]

IS_CERTIFIED = 'IsCertified'

IS_CONFIDENTIAL = 'Isconfidential'

REGISTRATION_DATE = 'RegistrationDate'

REGISTRATION_YEAR = 'RegistrationYear'

YEAR = 'year'

COUNTRY_COLUMN = 'Country'

COUNTRIES = [
             
             'IN',
             'BD',
             'CN',
             'SG',
             'MY',
             'ID',
             'JP',
             'LA',
             'KR',
             'VN'
             
             ]
CONVERT_TO_INT = 'PointsAchieved'

LABEL_ENCODER = [
                 

                'UnitOfMeasurement',
                'City',
                'ProjectTypes',
                'ProjectTypes2'
                 
                 
                 ]

TARGET_ENCODING = [
                 
                 None

                 ]

TARGET_COLUMN = 'CertLevel'

ONE_HOT_ENCODING = [
                    
                    'Country',
                    'ProjectTypes',
                    'OwnerTypes',
                    'OwnerTypes2'
                    
                    ]

STANDARDSCALER = ['GrossFloorArea','TotalPropArea']

NUMERICAL_ERROR = [
                    'GrossFloorArea',
                    'TotalPropArea'
                   ]




MODEL_DIR = "MODEL_DIR"
TRANSFORM_PICKLE_FILE_NAME = 'transformed_data.pkl'

TRAIN_TEST_SET_DATA_LOCATION = "ModelTrainingArtifacts"


#Model Building
MODEL_TRAINING_ARTIFACTS_DIR = "ModelTrainingArtifacts"

PARAM_GRID = {
    
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    # Add other parameters you want to tune

}


PARAM_GRID_RF = {
    
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']

}


