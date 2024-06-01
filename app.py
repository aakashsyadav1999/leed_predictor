import pandas as pd
import pickle
import os

from src.utils.common import save_object, load_object

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
preprocessor_path = os.path.join('MODEL_DIR','preprocessor.pkl')
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)


# Load the model from the pickle file
model_path = os.path.join('MODEL_DIR','model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the saved models and preprocessor
model_pkl_filepath = os.path.join('MODEL_DIR')
pickle_file_path = os.path.join(model_pkl_filepath, "model.pkl")
with open(pickle_file_path, "rb") as pickle_file:
    saved_object = pickle.load(pickle_file)


preprocessor = saved_object['preprocessor']
model = saved_object['model']


# Print the preprocessor to understand its structure
print(preprocessor)
print(model)

# Drop the target column 'CertLevel' from the sample data before transformation
sample_features = sample_df.drop(columns=['CertLevel'])

# Transform the sample data using the preprocessor
preprocessed_sample_data = preprocessor.transform(sample_df)

# Convert the transformed data back to a DataFrame if necessary
preprocessed_sample_data_df = pd.DataFrame(preprocessed_sample_data)
preprocessed_sample_data_df.to_csv("preprocessed_sample_data_df.csv")

# Save the preprocessed sample data to a CSV file
preprocessed_sample_data_df.to_csv('preprocessed_sample_data.csv', index=False)

# Predict using the loaded model
predictions = model.predict(preprocessed_sample_data_df)

# Combine predictions with the sample data
sample_df['Predictions'] = predictions

# Save the sample data with predictions to a CSV file
sample_df.to_csv('sample_data_with_predictions.csv', index=False)

print(sample_df)