import joblib
import numpy as np
import os


model_path = os.path.join(
    os.getenv("MODEL_OUTPUT_PATH", "./"), "linear_regression_model.pkl"
)
model = joblib.load(model_path)
print(model)

predictions = model.predict([[6], [7]])
print("Predictions for new data:", predictions)
