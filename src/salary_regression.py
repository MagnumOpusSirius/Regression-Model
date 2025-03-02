import pandas as pd
import numpy as np
import pickle
import dvc.api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


# Load dataset
data = pd.read_csv("salary_prediction_case_study.csv", encoding="ISO-8859-1")

# Preprocess dataset
data.fillna(data.select_dtypes(include=[np.number]).median(), inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].astype('category').cat.codes  # Encode categorical variables

#Feature selection
target = "Salary"
features = ["City", "Position", "GP"]
X = data[features]
y = data[target]

print("Feature names used in training:", X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R2: {r2}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
    
# Start an MLflow experiment
mlflow.set_experiment("Salary Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "salary_model")

    print(f"Model logged in MLflow: MAE={mae}, R2={r2}")
