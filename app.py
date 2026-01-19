app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("model/volatility_model.pkl")

app = FastAPI(title="TSLA Volatility Regime API")

# Input schema
class MarketFeatures(BaseModel):
    return_: float
    vol_10: float
    vol_20: float
    price_range: float
    volume_change: float

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_volatility(data: MarketFeatures):
    df = pd.DataFrame([{
        "return": data.return_,
        "vol_10": data.vol_10,
        "vol_20": data.vol_20,
        "price_range": data.price_range,
        "volume_change": data.volume_change
    }])

    prediction = model.predict(df)[0]

    regime_map = {
        0: "Low Volatility",
        1: "Medium Volatility",
        2: "High Volatility"
    }

    return {
        "volatility_regime": int(prediction),
        "description": regime_map[int(prediction)]
    }

