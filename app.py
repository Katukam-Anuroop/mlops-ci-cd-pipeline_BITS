from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os

app = Flask(__name__)

# Load the best model
MODEL_PATH = "deployment/model.npy"
theta = np.load(MODEL_PATH)

# Number of features used during training (TV, Radio, Newspaper)
NUM_FEATURES = len(theta) - 1  # Subtract bias term


@app.route("/")
def index():
    """
    Serve the HTML UI.
    """
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make predictions using the trained model.
    """
    try:
        # Get input data from request
        data = request.json
        features = data.get("features", [])

        # Validate input length
        if len(features) != NUM_FEATURES:
            return jsonify({"error": f"Expected {NUM_FEATURES} features, but got {len(features)}"}), 400

        # Convert input to numpy array
        X = np.array(features).reshape(1, -1)

        # Add bias term
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Make prediction
        prediction = X_with_bias.dot(theta).flatten()[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)