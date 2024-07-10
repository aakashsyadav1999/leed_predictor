from flask import Flask, request, jsonify
import logging
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return "Welcome to the LEED Prediction API!"

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    data = CustomData(
        City=str(request.json.get('City')),
        State=str(request.json.get('State')),
        Country=str(request.json.get('Country')),
        CertDate=str(request.json.get('CertDate')),
        OwnerTypes=str(request.json.get('OwnerTypes')),
        OwnerTypes2=str(request.json.get('OwnerTypes2')),
        GrossFloorArea=int(request.json.get('GrossFloorArea')),
        UnitOfMeasurement=str(request.json.get('UnitOfMeasurement')),
        TotalPropArea=float(request.json.get('TotalPropArea')),
        ProjectTypes=str(request.json.get('ProjectTypes')),
        ProjectTypes2=str(request.json.get('ProjectTypes2')),
        RegistrationDate=str(request.json.get('RegistrationDate')),
    )

    # Print the custom data
    logging.info(f"Received data: {data}")

    pred_df = data.get_data_as_data_frame()
    logging.info(f"DataFrame: \n{pred_df}")

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)[0]

    # Mapping results to labels
    mapping = {0: 'Denied', 1: 'Platinum', 2: 'Gold', 3: 'Certified', 4: 'Silver', 5: 'Bronze'}
    result_label = mapping.get(results, "Unknown")

    return jsonify({"results": result_label})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
