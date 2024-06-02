import pandas as pd
import pickle
import os

from src.utils.common import save_object, load_object

# Sample data
sample_data = {
    'City': ['Pittsburgh', 'Confidential', 'Chicago'],
    'State': ['PA', 'IN', 'IL'],
    'Country': ['IN', 'IN', 'IN'],
    'CertLevel': ['Certified','Certified','Certified'],
    'CertDate': ['01-10-2000 00:00', None, '05-11-2007 00:00'],
    'OwnerTypes': ['Corporate', 'Investor', 'Investor'],
    'OwnerTypes2': [' Publicly Traded', ' Equity Fund', 'Bank'],
    'GrossFloorArea': [647000, 291000, 22592],
    'UnitOfMeasurement': ['Sq ft', 'Sq ft', 'Sq ft'],
    'TotalPropArea': [202923, 130637, 27500],
    'ProjectTypes': ['Retail', 'Retail', 'Retail'],
    'ProjectTypes2': ['Open Shopping Center', 'Fast Food', 'Enclosed Mall'],
    'RegistrationDate': ['31-03-2000 00:00', '01-06-2000 00:00', '01-08-2001 00:00'],
    
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Display column names and types
print(sample_df.dtypes)


import pickle

model_path=os.path.join("MODEL_DIR","model.pkl")
preprocessor_path=os.path.join("MODEL_DIR","preprocessor.pkl")
print("Before Loading")

# Load the saved models and preprocessor
model_pkl_filepath = os.path.join('MODEL_DIR')
pickle_file_path = os.path.join(model_pkl_filepath, "model.pkl")
with open(pickle_file_path, "rb") as pickle_file:
    saved_object = pickle.load(pickle_file)


model = load_object(file_path=model_path)
preprocessor = load_object(file_path=preprocessor_path)


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