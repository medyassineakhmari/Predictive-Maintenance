# Predictive Maintenance and Anomaly Alerting System (MLOps)

This project implements an end-to-end predictive maintenance system designed for industrial applications. It leverages machine learning to process sensor data and forecast machine failures before they occur, reducing downtime and maintenance costs. 

## Project Overview

The core objective of this project is to provide a robust, production-ready machine learning pipeline. It transitions a predictive model from a standard notebook environment into a fully functional, containerized REST API, complete with continuous integration and continuous deployment (CI/CD) practices.

## Data Source

The model is trained on the AI4I 2020 Predictive Maintenance Dataset. This dataset reflects real-world industrial parameters, including air temperature, process temperature, rotational speed, torque, and tool wear. 

The dataset is publicly available on Kaggle and can be accessed here:
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

## System Architecture

The data flow and deployment strategy follow modern MLOps standards:

1. Machine Learning Model: The core engine uses a Random Forest classifier implemented via Scikit-Learn. It includes advanced handling of class imbalance and strict decision threshold optimization (set to 0.3) to prioritize the recall metric, ensuring critical failures are not missed.
2. Backend Deployment: The inference logic is served through an asynchronous REST API built with FastAPI. Data payload validation and typing are strictly enforced using Pydantic.
3. Containerization: The entire application, including the API and the serialized machine learning artifacts, is packaged using Docker to ensure environment consistency and cross-platform compatibility.
4. CI/CD Pipeline: Automated workflows are configured via GitHub Actions. Every code push or pull request triggers a pipeline that performs Python code linting, executes unit tests using Pytest, and verifies the Docker image build process.

## Execution Flow

[ IoT Sensor Data ] ---> [ FastAPI Endpoint ] ---> [ Random Forest Model ]
                                                               |
[ System Response ] <--- [ Status: NORMAL | ALERT | CRITICAL ] <

## Installation and Setup

### Method 1: Using Docker (Recommended)

Ensure Docker is installed and running on your system. 

1. Build the Docker image:
```bash
docker build -t predictive-maintenance-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 predictive-maintenance-api
```

The interactive API documentation (Swagger UI) will be accessible at: http://127.0.0.1:8000/docs

### Method 2: Using a Local Python Environment

Ensure Python 3.10 or higher is installed.

1. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the API server:
```bash
uvicorn main:app --reload
```

## API Usage Example

You can test the prediction endpoint using the interactive Swagger UI or by sending a direct POST request via cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Type": 1,
  "Air temperature [K]": 298.1,
  "Process temperature [K]": 308.6,
  "Rotational speed [rpm]": 1551.0,
  "Torque [Nm]": 42.8,
  "Tool wear [min]": 0.0
}'
```

---
*Developed by Mohammed-Yassine Akhmari - Data Science and Engineering*
