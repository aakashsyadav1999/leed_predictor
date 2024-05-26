import pandas as pd
import time
import numpy as np
from dataclasses import dataclass
import os
import stat

from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

@dataclass
#class for initiating all methods
class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.transformed_data = None  # To store the transformed DataFrame



    def read_csv(self):
        
        try:
            
            # Define the directory where the CSV files are stored
            data_directory = self.data_transformation_config.data_transformation_dir

            # List all files in the directory
            files = os.listdir(data_directory)
        
            # Filter for CSV files
            csv_files = [file for file in files if file.endswith('.csv')]

            # Check if there are any CSV files
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the directory")
            
            # Read the first CSV file (or handle as needed if there are multiple)
            csv_file_path = os.path.join(data_directory, csv_files[0])
            df = pd.read_csv(csv_file_path,encoding='latin1',low_memory=False)
            
            return df
        
        except Exception as e:
            logging.error(f"No CSV file found: {e}")
            raise
    

    def seprating_data(self,df):
        
        try:

            #selecting where is_certified is YES
            df.loc[(df[self.data_transformation_config.is_certified] == 'Yes')]

            #selecting where is_confidential is NO
            df = df.loc[(df[self.data_transformation_config.is_confidential] == 'No')]

            #Extracting year data from date column
            df.loc[:, self.data_transformation_config.registration_year] = pd.to_datetime(df[self.data_transformation_config.registration_date])

            #saving year for selecting data after 2015 data.
            df.loc[:, self.data_transformation_config.year] = pd.DatetimeIndex(df[self.data_transformation_config.registration_year]).year

            #Extracting data which is above 2015
            df = df.loc[(df[self.data_transformation_config.year] >= 2015)]

            #selecting data on country code
            df = df[df[self.data_transformation_config.country_column].isin(self.data_transformation_config.countries)]

            #Droping unwanted columns   
            df.drop(self.data_transformation_config.column_to_drop,inplace=True,axis=1)

            #drop NA
            df.dropna(inplace=True)

            self.transformed_data = df

        except Exception as e:
            raise e
    
    
    def save_to_pickle(self, directory, file_name):
        try:
            if self.transformed_data is None:
                raise ValueError("No transformed data to save. Run seprating_data method first.")

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Construct the file path
            file_path = os.path.join(directory, file_name)

            # Check if the directory is empty
            if os.path.exists(directory):
                # Delete all files in the directory
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    os.remove(file_path)

            # Save the transformed DataFrame to pickle file
            self.transformed_data.to_pickle(file_path)
            logging.info(f"Transformed data saved to {file_path}")

        except Exception as e:
            raise e


    #initiate all the methods which are mentioned above.
    def initiate_data_transformation(self):

        logging.info("Entered the initiate_data_transformation method of the data ingestion class")
        try:
            os.makedirs(
                self.data_transformation_config.data_transformation_dir,exist_ok=True
            )
            os.chmod(self.data_transformation_config.data_transformation_dir,stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"Creating {(self.data_transformation_config.data_transformation_dir)} directory")

            #check df
            df = self.read_csv()

            #seprating_data
            self.seprating_data(df)

            #saving pickle file
            self.save_to_pickle(self.data_transformation_config.model_dir,self.data_transformation_config.pickle_file_name)

        except Exception as e:
            raise e