import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("advertising_experiment")

# Ensure the 'models' directory exists
os.makedirs("models", exist_ok=True)

# Load dataset from CSV
data = pd.read_csv("advertising.csv")
data.fillna(data.mean(), inplace=True)
print(data.isnull().sum())

experiments = [
    {"features": ["TV"], "experiment_name": "Experiment_1_TV"},
    {"features": ["TV", "Radio"], "experiment_name": "Experiment_2_TV_Radio"},
    {"features": ["TV", "Radio", "Newspaper"],
     "experiment_name": "Experiment_3_All_Features"},
]


def gradient_descent(X, y, learning_rate=0.0001, iterations=1000):
    """
    Implements gradient descent for linear regression.
    """
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n + 1, 1))
    y = y.reshape(-1, 1)

    for i in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)

        # Clip gradients to prevent large updates
        gradients = np.clip(gradients, -1, 1)

        theta -= learning_rate * gradients

        if np.isnan(theta).any() or np.isinf(theta).any():
            print(f"Gradient Descent diverged at iteration {i}")
            break

    return theta


# Iterate through each experiment
for exp in experiments:
    print(f"Running {exp['experiment_name']} features: {exp['features']}")

    # Extract features and target variable
    scaler = StandardScaler()
    X = data[exp["features"]].values
    y = data["Sales"].values

    # Normalize features
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run(run_name=exp["experiment_name"]):
        # Train model using gradient descent
        theta = gradient_descent(
            X_train, y_train, learning_rate=0.0001, iterations=1000
        )

        # Make predictions
        X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        y_pred = X_test_with_bias.dot(theta).flatten()

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("features", ", ".join(exp["features"]))
        mlflow.log_param("method", "Gradient Descent")
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("iterations", 1000)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Save the model (theta values)
        model_path = f"models/{exp['experiment_name']}_model.npy"
        np.save(model_path, theta)
        mlflow.log_artifact(model_path)

        print(f"Model saved for {exp['experiment_name']} - R-squared: {r2}, MSE: {mse}")

print("All experiments completed.")
