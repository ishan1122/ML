from fastapi import FastAPI, Header, HTTPException
import joblib
import numpy as np

app = FastAPI()

# 🔐 API key
API_KEY = "fraudvision123"

# 🔥 Load models
xgb = joblib.load("models/xgboost.pkl")

# Optional
# lgbm = joblib.load("models/lightgbm.pkl")

@app.get("/")
def home():
    return {"status": "Fraud API Running"}

@app.post("/predict")
def predict(data: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    features = np.array(data["features"]).reshape(1, -1)

    prob = xgb.predict_proba(features)[0][1]

    return {
        "fraud_probability": float(prob),
        "prediction": int(prob > 0.1),
        "risk": (
            "Critical" if prob > 0.8 else
            "High" if prob > 0.6 else
            "Medium" if prob > 0.3 else
            "Low"
        )
    }