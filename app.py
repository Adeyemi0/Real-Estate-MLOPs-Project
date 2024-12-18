from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from real_estate.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Ensure the data is provided in the specified order
        data = CustomData(
            days_on_zillow=float(request.form.get('days_on_zillow')),
            zestimate=float(request.form.get('zestimate')),
            rent_zestimate=float(request.form.get('rent_zestimate')),
            area=float(request.form.get('area')),
            beds=float(request.form.get('beds')),
            baths=float(request.form.get('baths')),
            price_change=float(request.form.get('price_change')),
            tax_assessed_value=float(request.form.get('tax_assessed_value')),
            lot_area_value=float(request.form.get('lot_area_value')),
            home_type=request.form.get('home_type')
        )
        
        # Convert to DataFrame with ordered columns
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
