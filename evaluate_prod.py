import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load logs
preds = pd.read_csv("monitoring_log.csv")
feedback = pd.read_csv("feedback_log.csv")

# Merge on features
merged = pd.merge(
    preds,
    feedback,
    on=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    how="inner"
)

if merged.empty:
    print("No matched feedback yet.")
else:
    y_pred = merged["prediction"]
    y_true = merged["true_label"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=["setosa", "versicolor", "virginica"])

    print("=== Production Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
