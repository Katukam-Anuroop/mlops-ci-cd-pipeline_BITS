Linear Regression Prediction Application
This project provides a complete solution for predicting sales based on advertising budgets for TV, Radio, and Newspaper campaigns. It includes hyperparameter tuning, model selection, a user-friendly web interface, and containerization for deployment.

Features
Hyperparameter Tuning: Optimized learning rate and iterations for best model performance.
Model Training: Linear regression model trained on a real-world advertising dataset.
Web Interface: Simple, intuitive UI for entering advertising budgets and viewing predictions.
REST API: Flask-based API for handling prediction requests.
Containerization: Application packaged as a Docker container for easy deployment.
Project Structure
bash
Copy
Edit
.
├── app.py                # Flask application
├── index.html            # HTML interface for user interaction
├── model.npy             # Trained linear regression model
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # Project documentation
├── advertising.csv       # Dataset for training (tracked with DVC)
├── advertising.csv.dvc   # DVC metadata for dataset versioning
└── train.py              # Model training script with hyperparameter tuning
Setup and Usage
1. Clone the Repository
bash
Copy
Edit
git clone the repository

2. Install Dependencies
Create a virtual environment:
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
Install required packages:
bash
Copy
Edit
pip install -r requirements.txt
3. Train the Model
Run the train.py script to perform hyperparameter tuning and save the best model:

bash
Copy
Edit
python train.py
The script logs all experiments in MLflow for analysis.
4. Start the Application
Run the Flask app locally:

bash
Copy
Edit
python app.py
Open the app in your browser:
arduino
Copy
Edit
http://127.0.0.1:3000
5. Docker Containerization
Build the Docker Image
bash
Copy
Edit
docker build -t linear-regression-app .
Run the Docker Container
bash
Copy
Edit
docker run -p 5000:5000 linear-regression-app
Access the Application
Navigate to:
Copy
Edit
http://127.0.0.1:3000
API Reference
POST /predict
Endpoint for making predictions.

Request
json
Copy
Edit
{
    "features": [TV_budget, Radio_budget, Newspaper_budget]
}
Response
json
Copy
Edit
{
    "prediction": <predicted_sales>
}
Example Usage
Using the Web Interface
Open the app in your browser.
Enter advertising budgets for TV, Radio, and Newspaper.
Click Get Prediction to see the sales forecast.
Using curl
bash
Copy
Edit
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [100, 50, 20]}' \
http://127.0.0.1:3000/predict
Response:

json
Copy
Edit
{
    "prediction": 250.3
}
Technical Details
Hyperparameter Tuning
Parameters Tuned:
Learning Rate: [0.01, 0.001, 0.0001]
Iterations: [5000, 10000, 15000]
Best Configuration:
Features: TV, Radio, Newspaper
Learning Rate: 0.001
Iterations: 15000
R²: 0.9015
MSE: 3.109
Dataset
The dataset (advertising.csv) contains advertising budgets and sales data.
Tracked and versioned using DVC.
Tools and Frameworks
Python: Core programming language.
Flask: REST API and web framework.
NumPy: For mathematical computations.
Docker: For containerization.
MLflow: For experiment tracking.
DVC: For dataset versioning.
Future Enhancements
Cloud Deployment:
Deploy the Docker container on AWS/GCP/Azure.
CI/CD Pipeline:
Automate the container build and deployment process.
Scalability:
Add more advertising-related features to the dataset.
Contributors
Anuroop Katukam
Developer and Maintainer