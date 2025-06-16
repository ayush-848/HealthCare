import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define the root directory (one level up from the script's folder)
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Define paths for data and models relative to the root directory
data_path = os.path.join(root_dir, "data", "diabetes_prediction_dataset.csv")
models_dir = os.path.join(root_dir, "models")

# Load data
df = pd.read_csv(data_path)

# Drop rows with missing values (optional, or handle differently)
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)

# Features and label
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create models folder in the root directory
os.makedirs(models_dir, exist_ok=True)

# Save model, scaler, and feature order in the models folder
with open(os.path.join(models_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(models_dir, "feature_order.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, scaler, and feature order saved.")