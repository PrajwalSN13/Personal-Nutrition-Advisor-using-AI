# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from model import predict_food_suggestions

app = Flask(__name__)

# Load label encoders
label_encoders = {
    'Physical Activity Frequency': joblib.load('label_encoder_physical_activity.pkl'),
    'Chronic Kidney Disease': joblib.load('label_encoder_chronic_kidney_disease.pkl'),
    'Diabetes': joblib.load('label_encoder_diabetes.pkl'),
    'Allergy': joblib.load('label_encoder_allergy.pkl'),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.get_json()
        
        age = float(data['age'])
        weight = float(data['weight'])
        physical_activity = data['physical_activity']
        chronic_kidney_disease = data['chronic_kidney_disease']
        diabetes = data['diabetes']
        allergy = data['allergy']

        # Get predictions
        can_eat, cannot_eat = predict_food_suggestions(
            age, 
            weight, 
            physical_activity, 
            chronic_kidney_disease, 
            diabetes, 
            allergy
        )

        return jsonify({
            'success': True,
            'can_eat': can_eat,
            'cannot_eat': cannot_eat
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)