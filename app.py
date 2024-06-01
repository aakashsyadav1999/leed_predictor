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

# Load the training data to fit the preprocessor
train_df = pd.read_csv(r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\artifacts\DataTransformationArtifacts\train.csv', encoding='latin1', low_memory=False, skipinitialspace=True)

# Ensure 'CertLevel' is not in the preprocessing pipeline
if 'CertLevel' in train_df.columns:
    train_df = train_df.drop(columns=['CertLevel'])

# Fit the preprocessor on the training data
preprocessor.fit(train_df)

# Drop the target column 'CertLevel' from the sample data before transformation
sample_features = sample_df.drop(columns=['CertLevel'])

# Transform the sample data using the preprocessor
preprocessed_sample_data = preprocessor.transform(sample_features)

# Convert the transformed data back to a DataFrame if necessary
preprocessed_sample_data_df = pd.DataFrame(preprocessed_sample_data)

# Save the preprocessed sample data to a CSV file
preprocessed_sample_data_df.to_csv('preprocessed_sample_data.csv', index=False)

print(preprocessed_sample_data_df)