from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging
import pickle
import os
import pandas as pd
from dotenv import load_dotenv
from src.inference import make_prediction
import json


log_file_path = "logs/api_logs.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger("property-valuation-api")


app = FastAPI()
load_dotenv()


# Middleware for logging requests and responses
class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request details
        logger.info(
            f"Request path: {request.url.path} - Method: {request.method}"
        )

        # Process the request and get the response
        response = await call_next(request)

        # Log response status
        logger.info(f"Response status: {response.status_code}")

        return response


app.add_middleware(LogRequestsMiddleware)


# load champion model
async def load_model():
    with open("model/champion_model.pkl", "rb") as f:
        model = pickle.load(f)
        return model


class PropertyData(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


def verify_api_key(api_key: str):
    if api_key != os.getenv("API_KEY", default="NO_KEY_AVAILABLE"):
        logger.warning("Unauthorized access attempt with invalid API key")
        raise HTTPException(status_code=403, detail="Invalid API Key")


def add_demographics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add demographic data to the input DataFrame based on zipcode.
    """
    demographics = pd.read_csv("data/zipcode_demographics.csv")
    data = data.merge(demographics, on="zipcode", how="left")
    return data


def prepare_input_data(data: PropertyData) -> pd.DataFrame:
    """
    Convert PropertyData to DataFrame and add demographics.
    """
    model_features = json.load(open("model/champion_model_features.json", "r"))
    input_data = pd.DataFrame([data.dict()])
    input_data = add_demographics(input_data)
    input_data = input_data[model_features]

    return input_data


# Predict route
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(data: PropertyData):
    model = await load_model()
    input_data = prepare_input_data(data)

    prediction = make_prediction(input_data, model)
    return {"prediction": prediction[0]}


# Health Check
@app.get("/")
async def root():
    return {"message": "Property valuation API is running"}