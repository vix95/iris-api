import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import datetime
import pandas as pd
import os

# Load model and metadata
model = joblib.load("iris_model.pkl")
from sklearn.datasets import load_iris
iris = load_iris()
target_names = iris.target_names

# Request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Feedback schema
class Feedback(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    true_label: str   # e.g. "setosa", "versicolor", "virginica"

# Monitoring log file
LOG_FILE = "monitoring_log.csv"

# Feedback log file
FEEDBACK_FILE = "feedback_log.csv"

# Init app
app = FastAPI()

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "sepal_length", "sepal_width",
        "petal_length", "petal_width", "prediction"
    ]).to_csv(LOG_FILE, index=False)

@app.post("/predict")
def predict(features: IrisFeatures, request: Request):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(data)[0]
    label = target_names[prediction]

    # Log request
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width,
        "prediction": label
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    return {"prediction": label}

@app.post("/feedback")
def feedback(data: Feedback):
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
        "true_label": data.true_label
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    return {"status": "feedback received"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
