import pandas as pd


# Sample data
sample_data = {
    'ID': [10000000, 10000001, 10000002],
    'Isconfidential': ['No', 'No', 'No'],
    'ProjectName': ['PNC Firstside Center', 'Confidential', 'Bethel Commercial Center'],
    'Street': ['500 First Avenue', 'Confidential', '53 W. Jackson'],
    'City': ['Pittsburgh', 'Confidential', 'Chicago'],
    'State': ['PA', 'IN', 'IL'],
    'Zipcode': ['15219', None, '60604'],  # Handle missing values appropriately
    'Country': ['IN', 'IN', 'IN'],
    'LEEDSystemVersionDisplayName': ['LEED FOR SCHOOLS v2009', 'LEED FOR SCHOOLS v2009', 'LEED FOR SCHOOLS v2009'],
    'PointsAchieved': [33, 33, 45],
    'CertLevel': ['Certified','Certified','Certified'],
    'CertDate': ['01-10-2000 00:00', None, '05-11-2007 00:00'],
    'IsCertified': ['Yes', 'Yes', 'Yes'],
    'OwnerTypes': ['Corporate: Privately Held', 'Investor: Equity Fund', 'Investor: Bank'],
    'GrossFloorArea': [647000, 291000, 22592],
    'UnitOfMeasurement': ['Sq ft', 'Sq ft', 'Sq ft'],
    'TotalPropArea': [202923, 130637, 27500],
    'ProjectTypes': ['Retail', 'Retail', 'Retail'],
    'OwnerOrganization': ['L.D. Astorino Companies', 'Confidential', 'Bethel New Life'],
    'RegistrationDate': ['31-03-2000 00:00', '01-06-2000 00:00', '01-08-2001 00:00'],
    
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Display column names and types
print(sample_df.dtypes)


import pickle

# Load the preprocessor from the pickle file
preprocessor_path = r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\artifacts\preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Print the preprocessor to understand its structure
print(preprocessor)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Example column transformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['GrossFloorArea', 'TotalPropArea', 'PointsAchieved']),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), ['Isconfidential', 'City', 'State', 'Country'])
#     ]
# )

# # Example pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor)
# ])

# Transform the sample data using the preprocessor
train_df = pd.read_csv(r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\artifacts\DataTransformationArtifacts\train.csv',encoding='latin1', low_memory=False, skipinitialspace=True)
preprocessor.fit(train_df)
preprocessed_sample_data = preprocessor.transform(sample_df)

print(preprocessed_sample_data)
preprocessed_sample_data = pd.DataFrame(preprocessed_sample_data)
preprocessed_sample_data.to_csv('preprocessed_sample_data.csv')
