import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from optimize_emission import optimize_emissions

app = Flask(__name__)

model = joblib.load("model/carbon_emission_predictor.pkl")

def validate_input(value, name):
    """Ensure input is a valid non-negative number"""
    if value is None:
        return f"Missing input: {name} is required"
    try:
        num = float(value)  
        if num < 0:
            return f"Invalid input: {name} must be non-negative"
        return num
    except ValueError:
        return f"Invalid input: {name} must be a number"

@app.route('/')
def home():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON format"}), 400

        co = validate_input(data.get('co'), "CO")
        ch4 = validate_input(data.get('ch4'), "CH4")

        if isinstance(co, str) or isinstance(ch4, str):  
            return jsonify({"error": co if isinstance(co, str) else ch4}), 400  # Return validation error message

        input_data = pd.DataFrame([[co, ch4]], columns=['per capita CO (kg per person)', 'per capita CH4 (kg per person)'])
        predicted_co2 = model.predict(input_data)[0]

        return jsonify({"Predicted CO2 Emission": predicted_co2})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON format"}), 400

        initial_co = validate_input(data.get('co'), "CO")
        initial_ch4 = validate_input(data.get('ch4'), "CH4")

        if isinstance(initial_co, str) or isinstance(initial_ch4, str):  
            return jsonify({"error": initial_co if isinstance(initial_co, str) else initial_ch4}), 400

        optimized_result = optimize_emissions(initial_co, initial_ch4)
        return jsonify(optimized_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True)
