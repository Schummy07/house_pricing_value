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


# Load model at startup
with open("model/model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


class PropertyData(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float


def verify_api_key(api_key: str):
    if api_key != os.getenv("API_KEY", default="NO_KEY_AVAILABLE"):
        logger.warning("Unauthorized access attempt with invalid API key")
        raise HTTPException(status_code=403, detail="Invalid API Key")


# Predict route
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(data: PropertyData):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = make_prediction(input_data, model)
    return {"prediction": prediction[0]}


# Health Check
@app.get("/")
async def root():
    return {"message": "Property valuation API is running"}