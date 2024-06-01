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
                  ID: int,
                  Isconfidential: str,
                  ProjectName: str,
                  Street: str,
                  City: str,
                  State: str,
                  Zipcode: str,
                  Country: str,
                  LEEDSystemVersionDisplayName: str,
                  PointsAchieved: int,
                  CertDate:str,
                  IsCertified: str,
                  OwnerTypes: str,
                  GrossFloorArea: str,
                  UnitOfMeasurement: str,
                  TotalPropArea: int,
                  ProjectTypes:str,
                  OwnerOrganization: str,
                  RegistrationDate: str):
        
        self.ID = ID
        self.Isconfidential = Isconfidential
        self.ProjectName = ProjectName
        self.Street = Street
        self.City = City
        self.State = State
        self.Zipcode = Zipcode
        self.Country = Country
        self.LEEDSystemVersionDisplayName = LEEDSystemVersionDisplayName
        self.PointsAchieved = PointsAchieved
        self.CertDate = CertDate
        self.IsCertified = IsCertified
        self.OwnerType = OwnerTypes
        self.GrossFloorArea = GrossFloorArea
        self.UnitOfMeasurement = UnitOfMeasurement
        self.TotalPropArea = TotalPropArea
        self.ProjectTypes = ProjectTypes
        self.OwnerOrganization = OwnerOrganization
        self.RegistrationDate = RegistrationDate

    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "ID" : [self.ID],
                "Isconfidential" : [self.Isconfidential],
                "ProjectName" : [self.ProjectName],
                "Street" : [self.Street],
                "City" : [self.City],
                "State" : [self.State],
                "Zipcode" : [self.Zipcode],
                "Country" : [self.Country],
                "LEEDSystemVersionDisplayName" : [self.LEEDSystemVersionDisplayName],
                "PointsAchieved" : [self.PointsAchieved],
                "CertDate" : [self.CertDate],
                "IsCertified" : [self.IsCertified],
                "OwnerTypes" : [self.OwnerType],
                "GrossFloorArea" : [self.GrossFloorArea],
                "UnitOfMeasurement" : [self.UnitOfMeasurement],
                "TotalPropArea" : [self.TotalPropArea],
                "ProjectTypes" : [self.ProjectTypes],
                "OwnerOrganization" : [self.OwnerOrganization],
                "RegistrationDate" : [self.RegistrationDate]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise e