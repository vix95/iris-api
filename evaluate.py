import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import os

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load old model if exists
old_model_path = "iris_model.pkl"
old_accuracy = 0
if os.path.exists(old_model_path):
    old_model = joblib.load(old_model_path)
    old_pred = old_model.predict(X_test)
    old_accuracy = accuracy_score(y_test, old_pred)
    print(f"Old model accuracy: {old_accuracy:.2f}")

# Train new model
new_model = LogisticRegression(max_iter=200)
new_model.fit(X_train, y_train)
new_pred = new_model.predict(X_test)
new_accuracy = accuracy_score(y_test, new_pred)
print(f"New model accuracy: {new_accuracy:.2f}")

# Compare and decide
if new_accuracy >= old_accuracy:
    joblib.dump(new_model, "iris_model.pkl")
    print("✅ New model accepted and saved.")
else:
    print("❌ New model rejected (worse accuracy).")
