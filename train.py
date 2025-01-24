import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
X = np.array([[1], [2], [3], [4]])
y = np.array([2.5, 4.9, 7.4, 9.8])

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model

joblib.dump(model, "model.pkl")
print("Model trained and saved!")
