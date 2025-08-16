import pandas as pd

# Load the dataset
df = pd.read_csv("data/carbon_emissions.csv")

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Display dataset information
print("Dataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Save cleaned data
df.to_csv("data/cleaned_carbon_emissions.csv", index=False)
print("\nPreprocessing complete. Cleaned data saved.")

# Print the column names to verify
print("\nColumn Names in Dataset:")
print(df.columns)
