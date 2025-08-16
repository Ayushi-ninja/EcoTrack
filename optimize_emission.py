import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

# Load trained model
model = joblib.load("model/carbon_emission_predictor.pkl")

def predict_co2(co, ch4):
    """
    Predicts CO2 emissions based on CO and CH4 levels using the trained model.
    """
    input_data = pd.DataFrame([[co, ch4]], 
                              columns=['per capita CO (kg per person)', 'per capita CH4 (kg per person)'])
    return model.predict(input_data)[0]

def objective(x):
    """
    Objective function to minimize CO2 emissions.
    x[0] = CO (kg per person)
    x[1] = CH4 (kg per person)
    """
    return predict_co2(x[0], x[1])  # Minimize CO2 emissions

def constraint(x):
    """
    Constraint to ensure CO and CH4 remain within acceptable limits.
    """
    co_limit = 30  # Example: CO cannot exceed 30 kg per person
    ch4_limit = 25  # Example: CH4 cannot exceed 25 kg per person
    return [co_limit - x[0], ch4_limit - x[1]]  # Ensures x[0] <= 30 and x[1] <= 25

def optimize_emissions(initial_co, initial_ch4):
    """
    Optimizes CO and CH4 emissions to minimize CO2 emissions.
    """
    x0 = [initial_co, initial_ch4]  # Initial values for CO and CH4
    bounds = [(5, 30), (5, 25)]  # Lower and upper limits for CO and CH4
    constraints = {'type': 'ineq', 'fun': constraint}

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_co, optimized_ch4 = result.x
        optimized_co2 = predict_co2(optimized_co, optimized_ch4)
        return {
            "Optimized CO": optimized_co,
            "Optimized CH4": optimized_ch4,
            "Optimized CO2 Emission": optimized_co2
        }
    else:
        return {"error": "Optimization failed"}

# Example test
if __name__ == "__main__":
    result = optimize_emissions(20, 10)
    print(result)
