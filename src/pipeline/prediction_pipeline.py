import os
import sys
import pickle
import pandas as pd
from src.exception import NerException
from src.logger import logging
from src.utils.common import save_object,load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("MODEL_DIR","model.pkl")
            preprocessor_path=os.path.join("MODEL_DIR","preprocessor.pkl")
            print("Before Loading")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")
            print(f"Model type: {type(model)}")
            print(f"Preprocessor type: {type(preprocessor)}")

            if not hasattr(model, 'predict'):
                raise TypeError("Loaded model object does not have a 'predict' method")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        
        except Exception as e:
            raise NerException(e,sys)
        
class CustomData:

    def __init__ (self,
                  City: str,
                  State: str,
                  Country: str,
                  CertDate:str,
                  OwnerTypes: str,
                  OwnerTypes2: str,
                  GrossFloorArea: str,
                  UnitOfMeasurement: str,
                  TotalPropArea: int,
                  ProjectTypes:str,
                  ProjectTypes2:str,
                  RegistrationDate: str):
        

        self.City = City
        self.State = State
        self.Country = Country
        self.CertDate = CertDate
        self.OwnerTypes = OwnerTypes
        self.OwnerTypes2 = OwnerTypes2
        self.GrossFloorArea = GrossFloorArea
        self.UnitOfMeasurement = UnitOfMeasurement
        self.TotalPropArea = TotalPropArea
        self.ProjectTypes = ProjectTypes
        self.ProjectTypes2 = ProjectTypes2
        self.RegistrationDate = RegistrationDate

    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                
                "City" : [self.City],
                "State" : [self.State],
                "Country" : [self.Country],
                "CertDate" : [self.CertDate],
                "OwnerTypes" : [self.OwnerTypes],
                "OwnerTypes2" : [self.OwnerTypes2],
                "GrossFloorArea" : [self.GrossFloorArea],
                "UnitOfMeasurement" : [self.UnitOfMeasurement],
                "TotalPropArea" : [self.TotalPropArea],
                "ProjectTypes" : [self.ProjectTypes],
                "ProjectTypes2" : [self.ProjectTypes2],
                "RegistrationDate" : [self.RegistrationDate]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise e