import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained model
model = joblib.load("iris_model.pkl")

# Load iris target names for readable output
from sklearn.datasets import load_iris
iris = load_iris()
target_names = iris.target_names

# Define request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": target_names[prediction]}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
