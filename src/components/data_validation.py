import pickle
import pandas as pd

# Load the pre-trained models and scaler from the pickle file
model_scaler_path = r'D:\VS code files\upwork_work\Leed_Prediction\leed_predictor_end_end\MODEL_DIR\models_and_scalers.pkl'
with open(model_scaler_path, "rb") as file:
    models_and_scalers = pickle.load(file)

# Extract models and scaler from the loaded dictionary
xgb_model_cert = models_and_scalers["xgb_model_cert"]
xgb_model_points = models_and_scalers["xgb_model_points"]
rf_model_cert = models_and_scalers["rf_model_cert"]
scaler = models_and_scalers["scaler"]

def preprocess_input_data(input_data, scaler, columns_to_scale):
    """
    Preprocesses the input data using the loaded scaler.

    Args:
        input_data (pd.DataFrame): The input data to be preprocessed.
        scaler (StandardScaler): The scaler used for preprocessing.
        columns_to_scale (list): List of columns to be scaled.

    Returns:
        pd.DataFrame: The preprocessed input data.
    """
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
    return input_data

def predict(input_data):
    """
    Predicts the CertLevel and PointsAchieved using the pre-trained models.

    Args:
        input_data (pd.DataFrame): The input data for prediction.

    Returns:
        dict: The predicted CertLevel and PointsAchieved.
    """
    # Columns to be scaled (adjust as necessary)
    columns_to_scale = ["PointsAchieved", "GrossFloorArea", "TotalPropArea"]  # Adjust these columns as necessary

    # Preprocess the input data
    input_data_processed = preprocess_input_data(input_data, scaler, columns_to_scale)

    # Predict CertLevel using XGBoost classifier
    cert_predictions_xgb = xgb_model_cert.predict(input_data_processed)

    # Predict CertLevel using RandomForest classifier
    cert_predictions_rf = rf_model_cert.predict(input_data_processed)

    # Predict PointsAchieved using XGBoost regressor
    points_predictions = xgb_model_points.predict(input_data_processed)

    return {
        "CertLevel_XGB": cert_predictions_xgb,
        "CertLevel_RF": cert_predictions_rf,
        "PointsAchieved": points_predictions
    }

# Example usage
if __name__ == "__main__":
    # Create a DataFrame for the new input data from the given example (replace with actual user input data)
    new_data = pd.DataFrame({
        "Isconfidential": ["No"],
        "ProjectName": ["PNC Firstside Center"],
        "Street": ["500 First Avenue"],
        "City": ["Pittsburgh"],
        "State": ["PA"],
        "Zipcode": [15219],
        "Country": ["US"],
        "LEEDSystemVersionDisplayName": ["LEED-NC 2.0"],
        "PointsAchieved": [33],
        "CertLevel": ["Silver"],  # This is usually the target variable, but including it for completeness
        "CertDate": ["01-10-2000 00:00"],
        "IsCertified": ["Yes"],
        "OwnerTypes": ["Profit Org."],
        "GrossFloorArea": [647000],
        "UnitOfMeasurement": ["Sq ft"],
        "TotalPropArea": [202923],
        "ProjectTypes": ["Commercial Office"],
        "OwnerOrganization": ["L.D. Astorino Companies"],
        "RegistrationDate": ["31-03-2000 00:00"]
    })

    # Predict using the new input data
    predictions = predict(new_data)

    # Print the predictions
    print(predictions)