from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
import template

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            
            City=str(request.form.get('City')),
            State=str(request.form.get('State')),
            Country=str(request.form.get('Country')),
            CertDate=str(request.form.get('CertDate')),
            OwnerTypes=str(request.form.get('OwnerTypes')),
            OwnerTypes2=str(request.form.get('OwnerTypes2')),
            GrossFloorArea=int(request.form.get('GrossFloorArea')),
            UnitOfMeasurement=str(request.form.get('UnitOfMeasurement')),
            TotalPropArea=float(request.form.get('TotalPropArea')),
            ProjectTypes=str(request.form.get('ProjectTypes')),
            ProjectTypes2=str(request.form.get('ProjectTypes2')),
            RegistrationDate=str(request.form.get('RegistrationDate')),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
