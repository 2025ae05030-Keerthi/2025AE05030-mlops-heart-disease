import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops-service")

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

# Instrument Prometheus metrics endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

MODEL_PATH = "models/model_pipeline.joblib"
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    pipeline = None

class PatientData(BaseModel):
    age: float = Field(..., example=58.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=2.0)
    trestbps: float = Field(..., example=120.0)
    chol: float = Field(..., example=284.0)
    fbs: float = Field(..., example=0.0)
    restecg: float = Field(..., example=0.0)
    thalach: float = Field(..., example=160.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=1.8)
    slope: float = Field(..., example=2.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=3.0)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict")
def predict(data: PatientData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")
    
    input_df = pd.DataFrame([data.model_dump()])
    logger.info(f"Received inference request: {input_df.to_dict(orient='records')}")

    prediction = int(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    confidence = float(probabilities[prediction])

    result = {
        "prediction": prediction,
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": round(confidence, 4)
    }
    logger.info(f"Prediction result: {result}")
    return result