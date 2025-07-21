from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import pickle
import os
import pandas as pd
from dotenv import load_dotenv
from src.inference import make_prediction
from src.data_engineering import add_trend, add_house_age, add_renovation_flag
from xgboost import XGBRegressor
import json
import hashlib
import uuid
from datetime import datetime
from threading import Lock

# Configure logging
log_file_path = "logs/api_logs.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger("property-valuation-api")

# Environment
load_dotenv()
API_KEY = os.getenv("API_KEY", default="NO_KEY_AVAILABLE")

# FastAPI app
app = FastAPI()

# Globals for model caching & metadata
a_model = None
model_features = []
MODEL_CONSTRUCT = None
_model_mtime = None
_model_lock = Lock()


# Middleware for logging requests & responses
class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        payload_checksum = None
        try:
            body = await request.json()
            payload_checksum = hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()
        except Exception:
            pass
        logger.info(f"[RequestID={request_id}] Path={request.url.path} Method={request.method} Checksum={payload_checksum}")

        response = await call_next(request)
        logger.info(f"[RequestID={request_id}] Response status={response.status_code}")
        return response


app.add_middleware(LogRequestsMiddleware)

# Utility: load model & features from disk


def _load_model_from_disk():
    global a_model, model_features, MODEL_CONSTRUCT, _model_mtime
    model_path = "model/champion_model.pkl"
    feat_path = "model/champion_model_features.json"
    # Load model
    with open(model_path, "rb") as f:
        a_model = pickle.load(f)
    # Capture model construct as class name
    MODEL_CONSTRUCT = a_model.__class__.__name__
    # Load & sanitize features
    raw_feats = json.load(open(feat_path, "r"))
    model_features[:] = [f for f in raw_feats if isinstance(f, str) and f]
    # Update mtime
    _model_mtime = os.path.getmtime(model_path)
    logger.info(f"Loaded model construct={MODEL_CONSTRUCT} with {len(model_features)} features")


# On startup: initial model load
@app.on_event("startup")
async def startup_event():
    _load_model_from_disk()

# Ensure we have the latest champion before each request


def _ensure_latest_model():
    global _model_mtime
    try:
        current = os.path.getmtime("model/champion_model.pkl")
        if current != _model_mtime:
            with _model_lock:
                if current != _model_mtime:
                    _load_model_from_disk()
    except Exception as e:
        logger.error(f"Error checking model freshness: {e}")


# Data schemas
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


# API key verification
def verify_api_key(api_key: str):
    if api_key != API_KEY:
        logger.warning("Unauthorized access attempt with invalid API key")
        raise HTTPException(status_code=403, detail="Invalid API Key")


# Demographics merge
def add_demographics(data: pd.DataFrame) -> pd.DataFrame:
    demog = pd.read_csv("data/zipcode_demographics.csv")
    return data.merge(demog, on="zipcode", how="left")


# Feature engineering
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data['date'] = pd.to_datetime('2015-05-28')
    data = add_trend(data)
    data = add_house_age(data)
    data = add_renovation_flag(data)
    return data


# Prepare input data for prediction
def prepare_input_data(payload: PropertyData) -> pd.DataFrame:
    df = pd.DataFrame([payload.dict()])
    df = add_demographics(df)
    if isinstance(a_model, XGBRegressor):
        df = feature_engineering(df)
    available = [c for c in model_features if c in df.columns]
    missing = set(model_features) - set(available)
    if missing:
        logger.warning(f"Missing features: {missing}")
    return df[available]


# Prediction endpoint
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(data: PropertyData, request: Request):
    _ensure_latest_model()
    input_df = prepare_input_data(data)
    pred = make_prediction(input_df, a_model)[0]
    result = float(pred)

    prediction_time = datetime.utcnow().isoformat() + "Z"
    request_id = request.state.request_id
    input_checksum = hashlib.md5(json.dumps(data.dict(), sort_keys=True).encode()).hexdigest()

    return {
        "prediction": result,
        "model_construct": MODEL_CONSTRUCT,
        "prediction_time": prediction_time,
        "request_id": request_id,
        "input_checksum": input_checksum,
    }


# Health check
@app.get("/")
async def root():
    return {"message": "Property valuation API is running"}