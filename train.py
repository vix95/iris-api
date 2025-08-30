import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Run, Dataset

# Connect to Azure ML workspace
ws = Workspace.from_config()

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "iris_model.pkl")

# Register model in Azure ML
run = Run.get_context()
run.upload_file("outputs/iris_model.pkl", "iris_model.pkl")

model_reg = ws.models.register(
    model_path="iris_model.pkl",
    model_name="iris-classifier",
    description="RandomForest model for Iris classification"
)

print("Model registered:", model_reg.name, model_reg.version)
