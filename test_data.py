import pytest
from sklearn.datasets import load_iris
import numpy as np

def test_iris_data_shape():
    iris = load_iris()
    X = iris.data
    y = iris.target
    # X should have 4 features
    assert X.shape[1] == 4
    # Target should have same number of rows as X
    assert X.shape[0] == y.shape[0]

def test_iris_data_values():
    iris = load_iris()
    X = iris.data
    # All values should be finite numbers
    assert np.isfinite(X).all()
