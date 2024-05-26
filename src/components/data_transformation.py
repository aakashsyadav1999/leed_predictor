import pandas as pd
from dataclasses import dataclass
import os
import stat
import pickle

from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
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

            return df

        except Exception as e:
            raise e
    

    # Define the function for label encoding
    def label_encode_columns(self,df):
        
        try:
            
            le = LabelEncoder()
            
            for column in self.data_transformation_config.label_encoder:
                df[column] = le.fit_transform(df[column])
            
            return df
                
        except:
            pass

    # Target Encoding for high-cardinality features
    def target_encode(self, df, columns, target_column):

        try:

            for column in columns:

                encoder = TargetEncoder()
                
                df[column] = encoder.fit_transform(df[column], df[target_column])
            
            return df
            
        except Exception as e:
            raise e
        

    def one_hot_encoding(self,df):

        try:
            
            df = pd.get_dummies(
                                    
                                    df,
                                    columns=self.data_transformation_config.one_hot_encoding,
                                    dtype=int
                                    
                                    )
            df_head = df.head()

            return df

        except Exception as e:
            raise e
        
    def split_train_test_split(self,df):
        
        try:
            
            train,test = train_test_split(df,test_size=0.2,random_state=42)
            
            # Creating directories if they don't exist
            os.makedirs(self.data_transformation_config.train_test_file_path, exist_ok=True)

            # Saving train and test data to CSV files
            train.to_csv(os.path.join(self.data_transformation_config.train_test_file_path, 'train.csv'), index=False)
            test.to_csv(os.path.join(self.data_transformation_config.train_test_file_path, 'test.csv'), index=False)
            
            logging.info("Data splitted into train and test data.")
            logging.info(f'Shape of train data is {train.shape}')
            logging.info(f'Shape of test data is {test.shape}')
        
            return train, test
        
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
            df_sep = self.seprating_data(df)
            
            #label encoder
            df_label = self.label_encode_columns(df_sep)

            #target encoding
            df_encoded = self.target_encode(df_label,self.data_transformation_config.target_encoding,self.data_transformation_config.target_column)

            #one-hot-encoding
            df_final = self.one_hot_encoding(df_encoded)

            #splitting data into train and test
            self.split_train_test_split(df_final)

            # Splitting data into train and test
            train, test = self.split_train_test_split(df_final)

            # Create a dictionary to store train and test arrays
            data_dict = {"train": train.values, "test": test.values}

            # Create the directory if it doesn't exist
            os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)

            # Save the dictionary containing train and test arrays as a single pickle file
            pickle_file_path = os.path.join(
                self.data_transformation_config.model_dir, self.data_transformation_config.pickle_file_name
            )
            with open(pickle_file_path, "wb") as pickle_file:
                pickle.dump(data_dict, pickle_file)

            logging.info("Train and test arrays saved as a single pickle file.")

        except Exception as e:
            raise e