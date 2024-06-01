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
            ID=int(request.form.get('ID')),
            Isconfidential=str(request.form.get('Isconfidential')),
            ProjectName=str(request.form.get('ProjectName')),
            Street=str(request.form.get('Street')),
            City=str(request.form.get('City')),
            State=str(request.form.get('State')),
            Zipcode=int(request.form.get('Zipcode')),
            Country=str(request.form.get('Country')),
            LEEDSystemVersionDisplayName=str(request.form.get('LEEDSystemVersionDisplayName')),
            PointsAchieved=int(request.form.get('PointsAchieved')),
            CertDate=str(request.form.get('CertDate')),
            IsCertified=str(request.form.get('IsCertified')),
            OwnerTypes=str(request.form.get('OwnerTypes')),
            GrossFloorArea=int(request.form.get('GrossFloorArea')),
            UnitOfMeasurement=str(request.form.get('UnitOfMeasurement')),
            TotalPropArea=float(request.form.get('TotalPropArea')),
            ProjectTypes=str(request.form.get('ProjectTypes')),
            OwnerOrganization=str(request.form.get('OwnerOrganization')),
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
    app.run(host="0.0.0.0", port=5000)     