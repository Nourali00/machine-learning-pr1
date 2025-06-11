# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler from '{BASE_DIR}'. Make sure 'final_model.pkl' and 'scaler.pkl' are present. Original error: {e}")

# --- Define the exact feature names the model was trained on ---
# FIX: Added trailing spaces to 'FATIGUE' and 'ALLERGY' to exactly match
# the feature names the scaler/model was trained on.
FINAL_FEATURE_NAMES = [
    'GENDER', 'AGE', 'ANXIETY', 'FATIGUE ', 'ALLERGY ',
    'ALCOHOL CONSUMING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY',
    'CHEST PAIN', 'AGE_SMOKING_INTERACTION'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        age = int(data.get('AGE', 0))
        smoking = int(data.get('SMOKING', 0)) 
        data['AGE_SMOKING_INTERACTION'] = age * smoking

        # Create a DataFrame from the incoming JSON data
        input_df = pd.DataFrame([data])
        
        # Reorder and select columns to match FINAL_FEATURE_NAMES exactly.
        # This is the crucial step that ensures the data has the correct structure.
        input_features = input_df.reindex(columns=FINAL_FEATURE_NAMES, fill_value=0)

        features_scaled = scaler.transform(input_features)
        
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        response = {
            'prediction': int(prediction),
            'prediction_text': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'no_cancer': float(prediction_proba[0]),
                'cancer': float(prediction_proba[1])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Return a more detailed error message to help with debugging
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)