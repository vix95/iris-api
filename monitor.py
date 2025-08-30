import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import ks_2samp

# Load training data
iris = load_iris()
X_train = pd.DataFrame(iris.data, columns=iris.feature_names)

# Load production logs
logs = pd.read_csv("monitoring_log.csv")

# KS test (Kolmogorov-Smirnov) to check drift
for feature in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    stat, p_value = ks_2samp(X_train[feature], logs[feature])
    print(f"{feature}: p-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"⚠️ Potential drift detected in {feature}")
