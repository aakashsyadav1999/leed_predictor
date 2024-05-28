import pickle
import pandas as pd
import csv
import numpy as np


# Load the preprocessor from the pickle file
preprocessor_path = r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\MODEL_DIR\preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Load the model from the pickle file
model_pkl_path = r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\MODEL_DIR\models_and_scalers.pkl'
with open(model_pkl_path, 'rb') as f:
    model = pickle.load(f)

# Sample data
sample_data = {
    'ID': [10000000, 10000001, 10000002],
    'Isconfidential': ['No', 'No', 'No'],
    'ProjectName': ['PNC Firstside Center', 'Confidential', 'Bethel Commercial Center'],
    'Street': ['500 First Avenue', 'Confidential', '53 W. Jackson'],
    'City': ['Pittsburgh', 'Confidential', 'Chicago'],
    'State': ['PA', 'IN', 'IL'],
    'Zipcode': [15219, None, 60604],  # Handle missing values appropriately
    'Country': ['IN', 'IN', 'IN'],
    'LEEDSystemVersionDisplayName': ['LEED FOR SCHOOLS v2009', 'LEED FOR SCHOOLS v2009', 'LEED FOR SCHOOLS v2009'],
    'PointsAchieved': [33, None, 45],
    'CertDate': ['01-10-2000 00:00', None, '05-11-2007 00:00'],
    'IsCertified': ['Yes', 'Yes', 'Yes'],
    'OwnerTypes': ['Corporate: Privately Held', 'Investor: Equity Fund', 'Investor: Bank'],
    'GrossFloorArea': [647000, 291000, 22592],
    'UnitOfMeasurement': ['Sq ft', 'Sq ft', 'Sq ft'],
    'TotalPropArea': [202923, 130637, 27500],
    'ProjectTypes': ['Retail', 'Retail', 'Retail'],
    'OwnerOrganization': ['L.D. Astorino Companies', 'Confidential', 'Bethel New Life'],
    'RegistrationDate': ['31-03-2000 00:00', '01-06-2000 00:00', '01-08-2001 00:00']
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Transform the sample data using the preprocessor
preprocessed_sample_data = preprocessor.transform(sample_df)

print(preprocessed_sample_data)
#preprocessed_sample_data=pd.DataFrame(preprocessed_sample_data)
#preprocessed_sample_data.to_csv("preprocessed_sample.csv")
# Make predictions using the model
#predictions = model.predict(preprocessed_sample_data)

# # Add predictions to the DataFrame
#sample_df['Predicted_CertLevel'] = predictions

# # Print the DataFrame with predictions
#print(sample_df[['ID', 'CertLevel', 'Predicted_CertLevel']])