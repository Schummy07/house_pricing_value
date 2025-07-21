# House Pricing Model

This project implements a complete machine learning pipeline with champion/challenger models, including data engineering, training, model selection, and an API for inference.

## Prerequisites

- Python 3.9+
- Poetry (dependency manager)

## Environment Setup

### 1. Poetry Installation

First, install Poetry if you haven't already:

```bash
pip install poetry
```

### 1.a. Virtual Environment Creation and Activation

It's good practice to create an isolated virtual environment to keep dependencies organized:

```bash
poetry env activate
```

### 2. Dependency Installation

Install all project dependencies:

```bash
poetry install
```

## Initial Setup

### 3. Initial Champion Model Creation

Run the script to create the first champion model:

```bash
python create_model.py
```

This command will create the following files:
- `champion_model.pkl` - Serialized model
- `champion_model_features.json` - Model features
- `champion_model_metrics.json` - Performance metrics
- `champion_model_feature_importance.png` - Feature importance plot

### 3.a. Application Testing

After the initial champion model creation, it's already possible to test the endpoint and run the application. (Refer to API Usage)

### 3.b. API Key Configuration

Create a `.env` file in the project root and configure your API key:

```bash
# .env
API_KEY=<your_key>
```

**Important:** Replace `<your_key>` with your actual API key.

## Pipeline Execution

### 4. Full Pipeline Execution

Execute the main pipeline that will process data and train new models:

```bash
python pipeline.py
```

This command will execute the following modules:
- **data_engineering**: Data processing and preparation
- **train**: Training of new models
- **model_selection**: Model selection and comparison

#### Generated Artifacts:

- **Datasets**: Created in the `data/golden/`
- **Logs**: Stored in the `log/`
- **Challenger Model**: Saved in the `model/` with its artifacts
- **Automatic Replacement**: If the challenger model is superior to the current champion, it will be promoted to champion, and the previous model will be moved to `model/old_champion_version/`

## API Usage

### 5. Server Execution

Start the API server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 5.a. Programmatic Endpoint Testing

Run the test script to validate the endpoint with the first 5 data samples:

```bash
python test_endpoint.py
```

This script will use the first 5 examples from the `future_unseen_examples.csv` file to programmatically test the API.

### 5.b. Example API Request

You can use `curl` to send a request to the `/predict` endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: <YOUR_KEY>" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 1.0,
    "sqft_living": 1680,
    "sqft_lot": 5043,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 6,
    "sqft_above": 1680,
    "sqft_basement": 0,
    "yr_built": 1911,
    "yr_renovated": 0,
    "zipcode": 98118,
    "lat": 47.5354,
    "long": -122.273,
    "sqft_living15": 1560,
    "sqft_lot15": 5765
  }'
```

## API Endpoints

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

- Ensure the `.env` file is correctly configured
- Verify that all dependencies have been installed with `poetry install`
- Confirm that the champion model has been created before running the API
- Check the logs in the `log/` folder for errors during pipeline execution


