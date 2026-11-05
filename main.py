from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Model load karein
model = joblib.load("sk_ai_model.pkl")

# Data format define karein
class SignalData(BaseModel):
    signal_type: int
    wick_ratio: float
    body_size: float
    atr: float
    rsi: float
    volume: float

@app.get("/")
def home():
    return {"status": "AI Filter is Running Live!"}

@app.post("/predict")
def predict(data: SignalData):
    # Features array tayyar karna
    features = np.array([[
        data.signal_type,
        data.wick_ratio,
        data.body_size,
        data.atr,
        data.rsi,
        data.volume
    ]])
    
    # Model se win probability poochna
    win_prob = model.predict_proba(features)[0][1]
    
    # 70% confidence threshold (70% se zyada win chance ho toh allow karein)
    allow = bool(win_prob >= 0.70)
    
    return {
        "allow": allow,
        "confidence": round(float(win_prob), 2)
    }