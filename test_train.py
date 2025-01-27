import pytest
import numpy as np
from train import gradient_descent

def test_gradient_descent():
    # Simple dataset
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    # Run gradient descent
    theta = gradient_descent(X, y, learning_rate=0.01, iterations=100)
    
    # Assert the model converges to a close approximation
    assert theta[1][0] == pytest.approx(2, rel=0.1)
    assert theta[0][0] == pytest.approx(0, rel=0.1)
