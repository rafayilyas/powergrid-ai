from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import pandas as pd
from io import StringIO

# Your local imports
from app.schemas import DemandRequest, PeakRequest
from app.utils import load_models

app = FastAPI(title="Electricity Demand Prediction")


_models = None

def get_models():
    global _models
    if _models is None:
        _models = load_models()   # models load ONLY on first request
    return _models


# -----------------------------
# PREDICT ELECTRICITY DEMAND
# -----------------------------
@app.post("/predict-demand")
def predict_demand(req: DemandRequest):
    models = get_models()
    model = models.get("regression")

    if model is None:
        raise HTTPException(status_code=500, detail="Regression model not loaded")

    X = [[
        req.hour,
        req.temperature,
        req.voltage,
        req.dayofweek
    ]]

    prediction = model.predict(X)[0]

    return {
        "predicted_demand": float(prediction)
    }


# -----------------------------
# PEAK HOUR / LOAD SHEDDING RISK
# -----------------------------
@app.post("/peak-hour")
def peak_hour(req: PeakRequest):
    models = get_models()
    clf = models.get("classifier")

    if clf is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded")

    X = [[
        req.hour,
        req.temperature,
        req.voltage,
        req.dayofweek
    ]]

    y = clf.predict(X)[0]

    labels = {
        0: "Normal / Low Risk",
        1: "High Load Shedding Risk"
    }

    return {
        "risk": labels.get(int(y), "Unknown")
    }


# -----------------------------
# BULK CSV PREDICTION
# -----------------------------
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    models = get_models()
    model = models.get("regression")

    if model is None:
        raise HTTPException(status_code=500, detail="Regression model not loaded")

    try:
        # Read uploaded CSV
        contents = await file.read()
        csv_text = contents.decode("utf-8")

        df = pd.read_csv(StringIO(csv_text))

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ["hour", "temperature", "voltage", "dayofweek"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        # Ensure correct column order
        X = df[required_cols]

        predictions = model.predict(X)

        return {
            "predictions": predictions.tolist()
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")
