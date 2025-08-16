import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # For saving the trained model

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_carbon_emissions.csv")

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# ✅ Correct Feature Selection (X) and Target (y)
X = df[['per capita CO (kg per person)', 'per capita CH4 (kg per person)']]
y = df['per capita CO2 (kg per person)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, "model/carbon_emission_predictor.pkl")
print("\n✅ Model trained and saved successfully!")
