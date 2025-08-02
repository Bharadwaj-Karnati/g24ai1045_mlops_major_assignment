# MLOps Major Assignment

This project is a complete end-to-end MLOps pipeline using a Linear Regression model trained on the California Housing dataset. It demonstrates how to structure machine learning projects for reproducibility, testing, containerization (Docker), and automation via CI/CD using GitHub Actions.

---

## 📁 Project Structure

├── src/
│ ├── train.py # Train the linear regression model
│ ├── predict.py # Run model inference
│ ├── quantize.py # Quantize and evaluate model
│ └── utils.py # Utility functions (e.g., load data)
├── tests/
│ └── test_train.py # Unit tests for model and data
├── Dockerfile # Docker container setup
├── requirements.txt # Python dependencies
├── pytest.ini # Pytest configuration
├── .github/workflows/
│ └── ci.yml # GitHub Actions CI/CD pipeline
└── README.md # This file



---

## 🧠 Model Description

We use **Linear Regression** from `scikit-learn` to predict housing prices using the California housing dataset. The model is trained, evaluated, and then quantized to simulate compression for deployment.


---

## 🛠 Setup Instructions

### ✅ Clone the repository


git clone https://github.com/https://github.com/Bharadwaj-Karnati/g24ai1045_mlops_major_assignment.git
cd g24ai1045_mlops_major_assignment




✅ Create a virtual environment and activate it

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

✅ Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
🧪 Run the Code


✅ Train the model

python src/train.py


Output: 
(base) bk@ASUS2022:~/MLOPS Assignment$ python -m src.train
R2 score: 0.5758
Mean Squared Error: 0.5559


✅ Make predictions

python src/predict.py

(base) bk@ASUS2022:~/MLOPS Assignment$ python -m src.predict
   Predicted |     True
-------------------------
      0.7191 |   0.4770
      1.7640 |   0.4580
      2.7097 |   5.0000
      2.8389 |   2.1860
      2.6047 |   2.7800
      2.0118 |   1.5870
      2.6455 |   1.9820
      2.1688 |   1.5750
      2.7407 |   3.4000
      3.9156 |   4.4660

Model Evaluation Metrics:
R2 score: 0.5758
MSE: 0.5559

✅ Quantize and evaluate

python src/quantize.py

Output:

(base) bk@ASUS2022:~/MLOPS Assignment$ python -m src.quantize
R2 score with quantized params: 0.5758
MSE with quantized params: 0.5559


🐳 Docker Usage
✅ Build Docker image

docker build -t mlops-project .
✅ Run the Docker container

docker run --rm mlops-project

(base) bk@ASUS2022:~/MLOPS Assignment$ docker run --rm mlops-project
R2 score: 0.5758
Mean Squared Error: 0.5559

📦 The container runs src/predict.py by default.

✅ Run Tests

pytest
All tests are located in the tests/ directory and verify the model, data loading, and evaluation logic.

⚙️ GitHub Actions CI/CD
The CI/CD pipeline automatically:

Installs dependencies

Runs unit tests

Builds and runs the Docker image

Triggered on:
Push to main branch

Pull requests to main

File: .github/workflows/ci.yml