from dataclasses import dataclass
import os
import pandas as pd
from zipfile import ZipFile
from pathlib import Path

import zipfile
import stat
import gdown
from src.constants import *
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:

        self.data_ingestion_config = data_ingestion_config


    def download_file(self) -> str:
        '''
        Fetch data from the URL
        '''
        try:
            dataset_url = self.data_ingestion_config.source_url
            zip_download_dir = self.data_ingestion_config.local_data_file
            if not os.path.exists(self.data_ingestion_config.data_ingestion_artifacts_dir):
                os.makedirs(self.data_ingestion_config.data_ingestion_artifacts_dir)
                logging.info(f"Created directory: {self.data_ingestion_config.data_ingestion_artifacts_dir}")

            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logging.error(f"Error occurred during file download: {e}")
            raise e  
        
    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        try:
            zip_file_path = self.data_ingestion_config.local_data_file
            unzip_path = self.data_ingestion_config.unzip_csv_data

            if not os.path.exists(zip_file_path):
                raise FileNotFoundError(f"No such file or directory: '{zip_file_path}'")

            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info(f"Extracted file to directory: {unzip_path}")

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise e
        except zipfile.BadZipFile as e:
            logging.error(f"Bad zip file: {e}")
            raise e
        except Exception as e:
            logging.error(f"Error occurred during zip extraction: {e}")
            raise e
        
    def read_csv(self):
        try:
            data_directory = self.data_ingestion_config.data_ingestion_artifacts_dir
            files = os.listdir(data_directory)
            csv_files = [file for file in files if file.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the directory")
            csv_file_path = os.path.join(data_directory, csv_files[0])
            df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False, skipinitialspace=True)
            df['CertLevel'] = df['CertLevel'].replace({'Platinum':1,'Denied':0,'Gold':2,'Certified':3,'Silver':4,'Bronze':5})
            df.isnull().sum()
            return df
        except Exception as e:
            logging.error(f"No CSV file found: {e}")
            raise
        
    def split_train_test_split(self,df):
        try:
            train,test = train_test_split(df,test_size=0.2,random_state=42)

            train.to_csv(os.path.join(self.data_ingestion_config.split_train_test,'train.csv'),index=False)
            test.to_csv(os.path.join(self.data_ingestion_config.split_train_test,'test.csv'),index=False)
            
            logging.info("Data splitted into train and test data.")
            logging.info(f'Shape of train data is {train.shape}')
            logging.info(f'Shape of test data is {test.shape}')
        
            return train, test
        
        except Exception as e:
            logging.error(f"Error in split_train_test_data: {e}")
            raise


    def initiate_data_ingestion(self):
        logging.info("Entered the initiate_data_ingestion method of the data ingestion class")
        try:
            os.makedirs(self.data_ingestion_config.data_ingestion_artifacts_dir, exist_ok=True)
            os.chmod(self.data_ingestion_config.data_ingestion_artifacts_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"Set permissions and created directory: {self.data_ingestion_config.data_ingestion_artifacts_dir}")

            # Downloading data from given URL
            self.download_file()
            print("File Downloaded")
            logging.info(f"Downloaded data from URL: {self.data_ingestion_config.source_url}")

            # Extract file
            self.extract_zip_file()
            print("Extracting into Data_Transformation_Directory")
            logging.info(f"Extracted files into directory: {self.data_ingestion_config.unzip_csv_data}")

            df=self.read_csv()
            

            self.split_train_test_split(df)


            print("Extracted into Data_Transformation_Directory")
            logging.info('Extracted into Data_Transformation_Directory')
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion: {e}")
            raise e