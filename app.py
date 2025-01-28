from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.Predict_pipeline import CustomData,PredictPipeline

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
            Account_type = int(request.form.get('Account_type')),
            Duration_of_Credit_month = int(request.form.get('Duration_of_Credit_month')),
            Payment_Status_of_Previous_Credit = int(request.form.get('Payment_Status_of_Previous_Credit')),
            Purpose = int(request.form.get('Purpose')),
            Credit_Amount = int(request.form.get('Credit_Amount')),
            Savings_type = int(request.form.get('Savings_type')),
            Length_of_current_employment = int(request.form.get('Length_of_current_employment')),
            Instalment_percent = int(request.form.get('Instalment_percent')),
            Marital_Status = int(request.form.get('Marital_Status')),
            Guarantors = int(request.form.get('Guarantors')),
            Duration_in_Current_address = int(request.form.get('Duration_in_Current_address')),
            Most_valuable_available_asset = int(request.form.get('Most_valuable_available_asset')),
            Age = int(request.form.get('Age')),
            Concurrent_Credits = int(request.form.get('Concurrent_Credits')),
            Type_of_apartment = int(request.form.get('Type_of_apartment')),
            No_of_Credits_at_this_Bank = int(request.form.get('No_of_Credits_at_this_Bank')),
            Occupation = int(request.form.get('Occupation')),
            No_of_dependents = int(request.form.get('No_of_dependents')),
            Telephone = int(request.form.get('Telephone')),
            Foreign_Worker = int(request.form.get('Foreign_Worker'))
            
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)