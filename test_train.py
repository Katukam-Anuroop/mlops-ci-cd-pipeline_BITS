import pytest
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

def test_model_training():
    # Load the saved model
    model = joblib.load("model.pkl")
    assert isinstance(model, LinearRegression)
